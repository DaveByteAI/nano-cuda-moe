// An OpenAI-compatible HTTP server (one model, one request at a time):
//   GET  /v1/models
//   POST /v1/chat/completions   messages, stream, temperature, top_p, top_k, max_tokens / max_completion_tokens, seed,
//                               reasoning_effort, chat_template_kwargs.enable_thinking; non-standard: spec (MTP drafts)
//   GET  /health
// Function calling: "tools" go into the model's template; the model's <tool_call> blocks come back as "tool_calls"
// (finish_reason "tool_calls"; streamed, the text before them streams and the calls come in one delta at the end), and
// assistant turns with "tool_calls" and "tool" messages are written back the template's way.
// Thinking is off unless the request asks (enable_thinking: true or a reasoning_effort); the thinking comes back as
// reasoning_content. The engine keeps the conversation between requests: a request that continues it (the same
// messages plus new ones) only feeds the new part.
//   bl-server --model SHARD1.gguf [--host 127.0.0.1] [--port 8080] [--ctx 32k|64k|128k|256k] [--name NAME] [--web DIR|none]
//             [--expert-cache FILE]
// Switching models while running: GET /bl/models lists the quantizations next to --model (files "...-00001-of-N.gguf"
// with all their shards) and the state; POST /bl/models {"id": "IQ3_S"} answers at once and swaps the model in the
// background once the current reply is done (chat requests meanwhile get 503); if the new one fails to load, the old
// one comes back. The chat page has it in its settings panel.
// The chat page (web/index.html) is served at /. With --expert-cache, the expert cache (which experts sit in VRAM) is
// saved after every reply and restored at the next start. Off by default: on new text it did not raise the hit rate
// (88.4% restored vs 89.0% from the profile, 2026-10-05).
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "bl/chat.h"
#include "httplib.h"
#include "nlohmann/json.hpp"

using json = nlohmann::ordered_json;

namespace {

std::string content_text(const json & c) {   // a message's content: a string, or parts of which the text ones count
    if (c.is_string()) return c.get<std::string>();
    std::string s;
    if (c.is_array())
        for (const auto & part : c)
            if (part.is_object() && part.value("type", "") == "text") s += part.value("text", "");
    return s;
}

// a model file's id: the quantization in "...GSQ-RCO-<id>-00001-of-N.gguf", else the name before "-00001-of-"
std::string model_id(const std::string & path) {
    const std::string f = path.substr(path.rfind('/') + 1);
    const size_t end = f.find("-00001-of-"), at = f.rfind("GSQ-RCO-", end);
    if (end == std::string::npos) return f;
    return at == std::string::npos ? f.substr(0, end) : f.substr(at + 8, end - at - 8);
}

struct ModelFile { std::string id, path; double gb; };
// the models in shard 1's directory whose shards are all there (sizes through symlinks)
std::vector<ModelFile> list_models(const std::string & shard1) {
    const size_t sl = shard1.rfind('/');
    const std::string dir = sl == std::string::npos ? "." : shard1.substr(0, sl);
    std::vector<ModelFile> r;
    DIR * d = opendir(dir.c_str());
    if (!d) return r;
    while (const dirent * e = readdir(d)) {
        const std::string f = e->d_name;
        const size_t at = f.find("-00001-of-");
        if (at == std::string::npos || f.size() < 5 || f.compare(f.size() - 5, 5, ".gguf") != 0) continue;
        const int n = std::atoi(f.c_str() + at + 10);
        bool all = n >= 1;
        double gb = 0;
        for (int k = 1; all && k <= n; ++k) {
            char part[16];
            std::snprintf(part, sizeof part, "-%05d-of-", k);
            std::string g = f;
            g.replace(at, 10, part);
            struct stat st {};
            all = ::stat((dir + "/" + g).c_str(), &st) == 0;
            gb += st.st_size / 1e9;
        }
        if (all) r.push_back({model_id(f), dir + "/" + f, gb});
    }
    closedir(d);
    std::sort(r.begin(), r.end(), [](const ModelFile & a, const ModelFile & b) { return a.id < b.id; });
    return r;
}

// JSON as the template's tojson writes it (Python's json.dumps: ", " and ": " between items, no ASCII escaping)
std::string py_json(const json & v) {
    if (v.is_object()) {
        std::string s = "{";
        for (auto it = v.begin(); it != v.end(); ++it) s += (it == v.begin() ? "" : ", ") + json(it.key()).dump() + ": " + py_json(it.value());
        return s + "}";
    }
    if (v.is_array()) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + py_json(v[i]);
        return s + "]";
    }
    return v.dump();
}

std::string trim_ws(const std::string & s) {
    const auto b = s.find_first_not_of(" \t\r\n"), e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}

// the answer split into its text and its <tool_call> blocks, as OpenAI tool_calls; a parameter's value is a string
// when the tool's schema says so (or it is not JSON), else the JSON it holds
struct Answer { std::string content; json calls = json::array(); };
Answer split_tool_calls(const std::string & text, const json & tools, const std::string & id_prefix) {
    Answer a;
    size_t at = text.find("<tool_call>");
    a.content = trim_ws(text.substr(0, at));
    while (at != std::string::npos) {
        const size_t end = text.find("</tool_call>", at);
        const std::string blk = text.substr(at, end == std::string::npos ? std::string::npos : end - at);
        const size_t f = blk.find("<function=");
        if (f != std::string::npos) {
            const size_t fe = blk.find('>', f);
            const std::string name = trim_ws(blk.substr(f + 10, fe - f - 10));
            json props = json::object();
            for (const auto & t : tools) {
                const json & fn = t.contains("function") ? t["function"] : t;
                if (fn.value("name", "") == name && fn.contains("parameters")) props = fn["parameters"].value("properties", json::object());
            }
            json args = json::object();
            for (size_t p = blk.find("<parameter=", fe); p != std::string::npos; p = blk.find("<parameter=", p + 1)) {
                const size_t pe = blk.find('>', p), ve = blk.find("</parameter>", pe);
                if (pe == std::string::npos) break;
                const std::string key = trim_ws(blk.substr(p + 11, pe - p - 11));
                std::string val = blk.substr(pe + 1, ve == std::string::npos ? std::string::npos : ve - pe - 1);
                if (!val.empty() && val.front() == '\n') val.erase(0, 1);
                if (!val.empty() && val.back() == '\n') val.pop_back();
                const bool as_string = props.contains(key) && props[key].value("type", "") == "string";
                json v = val;
                if (!as_string) try { v = json::parse(val); } catch (...) { v = val; }
                args[key] = v;
            }
            a.calls.push_back({{"id", id_prefix + std::to_string(a.calls.size())}, {"type", "function"},
                               {"function", {{"name", name}, {"arguments", args.dump()}}}});
        }
        at = end == std::string::npos ? std::string::npos : text.find("<tool_call>", end);
    }
    return a;
}

json error_json(const std::string & msg, const std::string & type = "invalid_request_error") {
    return json{{"error", {{"message", msg}, {"type", type}}}};
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model, host = "127.0.0.1", name, web, cache_file;   // name: from the model's id unless given
    int port = 8080, ctx = 32768;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--model") model = next();
        else if (a == "--host") host = next();
        else if (a == "--port") port = std::stoi(next());
        else if (a == "--ctx") ctx = bl::parse_ctx(next());
        else if (a == "--name") name = next();
        else if (a == "--web") web = next();   // the chat page's directory (default: ../web next to the executable)
        else if (a == "--expert-cache") cache_file = next();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model.empty()) { std::fprintf(stderr, "usage: %s --model SHARD1.gguf [--host H] [--port P] [--ctx N]\n", argv[0]); return 2; }

    // the loaded model; replaced (under `busy`) by a switch
    struct Loaded { std::unique_ptr<bl::Text> text; std::unique_ptr<bl::Engine> eng; std::unique_ptr<bl::Chat> chat; std::string path; };
    Loaded cur;
    const bool fixed_name = !name.empty();
    std::mutex state_mu;   // guards the switch state below and the model's name (read without waiting for `busy`)
    std::string current, target, state = "ready", switch_error;
    std::atomic<bool> switching{false};
    auto model_name = [&] { std::lock_guard<std::mutex> lk(state_mu); return name; };
    auto load = [&](const std::string & path) {   // the old one goes first: two do not fit
        cur.chat.reset();
        cur.eng.reset();
        cur.text.reset();
        std::fprintf(stderr, "loading %s ...\n", path.c_str());
        cur.text = std::make_unique<bl::Text>(path);
        cur.eng = std::make_unique<bl::Engine>(path, ctx, cache_file);
        cur.chat = std::make_unique<bl::Chat>(*cur.eng, *cur.text, ctx);
        cur.path = path;
        std::lock_guard<std::mutex> lk(state_mu);
        current = model_id(path);
        if (!fixed_name) {
            name = "qwen3.8-flash-" + current;
            std::transform(name.begin(), name.end(), name.begin(), [](unsigned char ch) { return std::tolower(ch); });
        }
    };
    load(model);
    auto save_cache = [&] {   // after a reply (under `busy`): the next start begins with this cache
        if (cache_file.empty() || !cur.eng) return;
        try { cur.eng->save_cache(cache_file); } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); }
    };
    std::mutex busy;   // one generation (or a model switch) at a time
    std::atomic<long> seq{0};

    httplib::Server srv;
    if (web.empty()) {   // ../web relative to this executable (build/bl-server -> web/)
        char exe[4096];
        const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
        if (n > 0) {
            std::string d(exe, static_cast<size_t>(n));
            d = d.substr(0, d.rfind('/'));
            web = d.substr(0, d.rfind('/')) + "/web";
        }
    }
    if (web != "none" && srv.set_mount_point("/", web)) std::fprintf(stderr, "chat page: %s\n", web.c_str());
    srv.Get("/health", [](const httplib::Request &, httplib::Response & res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
    srv.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"object", "list"}, {"data", json::array({{{"id", model_name()}, {"object", "model"}, {"owned_by", "boundless"}}})}}.dump(),
                        "application/json");
    });
    srv.Get("/bl/models", [&](const httplib::Request &, httplib::Response & res) {
        json list = json::array();
        for (const auto & m : list_models(model)) list.push_back({{"id", m.id}, {"gb", std::round(m.gb * 10) / 10}});
        std::lock_guard<std::mutex> lk(state_mu);
        res.set_content(json{{"current", current}, {"state", state}, {"target", target}, {"error", switch_error}, {"models", list}}.dump(),
                        "application/json");
    });
    srv.Post("/bl/models", [&](const httplib::Request & req, httplib::Response & res) {
        std::string id;
        try { id = json::parse(req.body).at("id").get<std::string>(); }
        catch (...) { res.status = 400; res.set_content(error_json("expected {\"id\": \"<quantization>\"}").dump(), "application/json"); return; }
        std::string path;
        for (const auto & m : list_models(model)) if (m.id == id) path = m.path;
        if (path.empty()) { res.status = 404; res.set_content(error_json("no model " + id).dump(), "application/json"); return; }
        {
            std::lock_guard<std::mutex> lk(state_mu);
            if (switching) { res.status = 409; res.set_content(error_json("already switching to " + target).dump(), "application/json"); return; }
            if (id == current && state == "ready") { res.set_content(json{{"state", "ready"}, {"current", current}}.dump(), "application/json"); return; }
            switching = true;
            state = "loading";
            target = id;
            switch_error.clear();
        }
        std::thread([&, path, id] {
            std::lock_guard<std::mutex> lk(busy);   // after the current reply
            const auto t0 = std::chrono::steady_clock::now();
            const std::string prev = cur.path;
            std::string err;
            std::fprintf(stderr, "switching to %s\n", id.c_str());
            try { load(path); } catch (const std::exception & e) { err = e.what(); }
            if (!err.empty()) {   // the old one back
                std::fprintf(stderr, "switch to %s failed: %s; reloading %s\n", id.c_str(), err.c_str(), prev.c_str());
                try { load(prev); } catch (const std::exception & e) { err += std::string("; reloading the previous model failed too: ") + e.what(); }
            } else {
                std::fprintf(stderr, "switched to %s in %.1f s\n", id.c_str(),
                             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            }
            std::lock_guard<std::mutex> sl(state_mu);
            state = cur.chat ? "ready" : "failed";
            switch_error = err;
            switching = false;
        }).detach();
        res.status = 202;
        res.set_content(json{{"state", "loading"}, {"target", id}}.dump(), "application/json");
    });
    srv.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        if (switching) {
            res.status = 503;
            std::lock_guard<std::mutex> lk(state_mu);
            res.set_content(error_json("switching the model to " + target + ", try again in a minute", "server_busy").dump(), "application/json");
            return;
        }
        const std::string name = model_name();
        json body;
        try { body = json::parse(req.body); } catch (...) { res.status = 400; res.set_content(error_json("invalid JSON").dump(), "application/json"); return; }
        std::vector<bl::Text::Message> msgs;
        bl::ChatParams cp;
        try {
            for (const auto & m : body.at("messages")) {
                bl::Text::Message msg{m.at("role").get<std::string>(), content_text(m.value("content", json(""))),
                                      m.value("reasoning_content", json("")).is_string() ? m.value("reasoning_content", std::string()) : ""};
                if (m.contains("tool_calls") && m["tool_calls"].is_array())
                    for (const auto & tc : m["tool_calls"]) {   // arguments: a JSON string (OpenAI) or an object
                        const json & fn = tc.contains("function") ? tc["function"] : tc;
                        bl::Text::ToolCall call{fn.at("name").get<std::string>(), {}};
                        json args = fn.value("arguments", json::object());
                        if (args.is_string()) args = trim_ws(args.get<std::string>()).empty() ? json::object() : json::parse(args.get<std::string>());
                        for (auto it = args.begin(); it != args.end(); ++it)
                            call.args.push_back({it.key(), it.value().is_string() ? it.value().get<std::string>() : py_json(it.value())});
                        msg.tool_calls.push_back(std::move(call));
                    }
                msgs.push_back(std::move(msg));
            }
            if (body.contains("tools") && body["tools"].is_array())
                for (const auto & t : body["tools"]) cp.tools.push_back(py_json(t));
            cp.sampling.temperature = body.value("temperature", 0.7f);
            cp.sampling.top_p = body.value("top_p", 0.8f);
            cp.sampling.top_k = body.value("top_k", 20);
            cp.sampling.seed = body.value("seed", 0ull);
            cp.max_new = body.value("max_completion_tokens", body.value("max_tokens", 4096));
            cp.spec = body.value("spec", 3);
            const bool think = body.contains("chat_template_kwargs") && body["chat_template_kwargs"].value("enable_thinking", false);
            std::string effort = body.value("reasoning_effort", std::string());
            if (effort == "high") effort = "xhigh";
            cp.think = !effort.empty() ? effort : think ? "xhigh" : "";
        } catch (const std::exception & e) {
            res.status = 400;
            res.set_content(error_json(e.what()).dump(), "application/json");
            return;
        }
        const bool stream = body.value("stream", false);
        const json tools = body.value("tools", json::array());
        const std::string id = "chatcmpl-" + std::to_string(++seq);
        const long created = static_cast<long>(std::time(nullptr));
        auto usage = [](const bl::ChatResult & r) {
            return json{{"prompt_tokens", r.n_prompt + r.n_reused}, {"completion_tokens", r.n_generated},
                        {"total_tokens", r.n_prompt + r.n_reused + r.n_generated},
                        {"prompt_tokens_details", {{"cached_tokens", r.n_reused}}}};
        };
        if (!stream) {
            std::lock_guard<std::mutex> lk(busy);
            try {
                if (!cur.chat) throw std::runtime_error("no model loaded (the last switch failed)");
                const bl::ChatResult r = cur.chat->reply(msgs, cp);
                const Answer a = tools.empty() ? Answer{r.content} : split_tool_calls(r.content, tools, "call_" + std::to_string(seq) + "_");
                json msg{{"role", "assistant"}, {"content", a.content.empty() && !a.calls.empty() ? json(nullptr) : json(a.content)}};
                if (!r.reasoning.empty()) msg["reasoning_content"] = r.reasoning;
                if (!a.calls.empty()) msg["tool_calls"] = a.calls;
                const char * finish = !a.calls.empty() ? "tool_calls" : r.stopped_eog ? "stop" : "length";
                res.set_content(json{{"id", id}, {"object", "chat.completion"}, {"created", created}, {"model", name},
                                     {"choices", json::array({{{"index", 0}, {"message", msg}, {"finish_reason", finish}}})},
                                     {"usage", usage(r)}}.dump(), "application/json");
                std::fprintf(stderr, "%s: %d prompt (+%d reused) in %.2f s, %d tokens at %.1f tok/s\n", id.c_str(), r.n_prompt, r.n_reused,
                             r.prompt_s, r.n_generated, r.gen_s > 0 ? (r.n_generated - 1) / r.gen_s : 0.0);
                save_cache();
            } catch (const std::exception & e) {
                res.status = 400;
                res.set_content(error_json(e.what()).dump(), "application/json");
            }
            return;
        }
        res.set_chunked_content_provider("text/event-stream", [&, msgs, cp, id, created, name, tools](size_t, httplib::DataSink & sink) {
            std::lock_guard<std::mutex> lk(busy);
            auto send = [&](const json & j) {
                const std::string s = "data: " + j.dump() + "\n\n";
                return sink.write(s.data(), s.size());
            };
            auto chunk = [&](const json & delta, const json & finish) {
                return json{{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", name},
                            {"choices", json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})}};
            };
            send(chunk({{"role", "assistant"}, {"content", ""}}, nullptr));
            try {
                if (!cur.chat) throw std::runtime_error("no model loaded (the last switch failed)");
                // with tools, the text streams up to a <tool_call> (a tail that could be its start waits); the calls follow at the end
                static const std::string kTag = "<tool_call>";
                std::string held;
                bool in_call = false;
                auto text_out = [&](const std::string & s) { return s.empty() || send(chunk(json{{"content", s}}, nullptr)); };
                const bl::ChatResult r = cur.chat->reply(msgs, cp, [&](const std::string & s, bool reasoning) {
                    if (reasoning) return send(chunk(json{{"reasoning_content", s}}, nullptr));   // false: the client left
                    if (tools.empty()) return text_out(s);
                    if (in_call) return true;
                    held += s;
                    if (const size_t at = held.find(kTag); at != std::string::npos) {
                        in_call = true;
                        return text_out(held.substr(0, at));
                    }
                    size_t keep = std::min(held.size(), kTag.size() - 1);
                    while (keep && held.compare(held.size() - keep, keep, kTag, 0, keep) != 0) --keep;
                    const std::string out = held.substr(0, held.size() - keep);
                    held.erase(0, held.size() - keep);
                    return text_out(out);
                });
                if (!in_call) text_out(held);
                const Answer a = tools.empty() ? Answer{} : split_tool_calls(r.content, tools, "call_" + id.substr(9) + "_");
                if (!a.calls.empty()) {
                    json tc = a.calls;
                    for (size_t i = 0; i < tc.size(); ++i) tc[i]["index"] = static_cast<int>(i);
                    send(chunk(json{{"tool_calls", tc}}, nullptr));
                }
                json last = chunk(json::object(), !a.calls.empty() ? "tool_calls" : r.stopped_eog ? "stop" : "length");
                last["usage"] = usage(r);
                send(last);
                std::fprintf(stderr, "%s (stream): %d prompt (+%d reused) in %.2f s, %d tokens at %.1f tok/s\n", id.c_str(), r.n_prompt,
                             r.n_reused, r.prompt_s, r.n_generated, r.gen_s > 0 ? (r.n_generated - 1) / r.gen_s : 0.0);
                save_cache();
            } catch (const std::exception & e) {
                send(error_json(e.what()));
            }
            const std::string done = "data: [DONE]\n\n";
            sink.write(done.data(), done.size());
            sink.done();
            return true;
        });
    });
    std::fprintf(stderr, "listening on http://%s:%d (OpenAI API: /v1/chat/completions)\n", host.c_str(), port);
    if (!srv.listen(host, port)) { std::fprintf(stderr, "cannot listen on %s:%d\n", host.c_str(), port); return 1; }
    return 0;
}
