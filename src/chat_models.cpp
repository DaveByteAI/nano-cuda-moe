// The ChatModel implementations: qwen3.8 (Engine + Chat as they are) and deepseek41 (Dsv41Engine; the template from
// the GGUF's tokenizer.chat_template written out, the session kept between turns, sampling on the host).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>

#include "bl/chat.h"
#include "bl/dsv41.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <unistd.h>

namespace bl {

namespace {

std::string architecture(const std::string & path) {
    gguf_init_params ip{true, nullptr};
    gguf_context * g = gguf_init_from_file(path.c_str(), ip);
    if (!g) throw std::runtime_error("cannot read " + path);
    const int64_t k = gguf_find_key(g, "general.architecture");
    const std::string a = k >= 0 ? gguf_get_val_str(g, k) : "";
    gguf_free(g);
    return a;
}

class QwenChat : public ChatModel {
public:
    QwenChat(const std::string & path, int ctx, const std::string & cache) : text_(path), eng_(path, ctx, cache), chat_(eng_, text_, ctx), ctx_(ctx) {}
    ChatResult reply(const std::vector<Text::Message> & m, const ChatParams & p,
                     const std::function<bool(const std::string &, bool)> & on_text) override { return chat_.reply(m, p, on_text); }
    void        reset() override { chat_.reset(); }
    int         position() const override { return chat_.position(); }
    int         max_ctx() const override { return ctx_; }
    std::string report() const override { return eng_.report(); }
    std::string family() const override { return "qwen3.8-flash"; }
    void        save_cache(const std::string & f) override { eng_.save_cache(f); }
private:
    Text   text_;
    Engine eng_;
    Chat   chat_;
    int    ctx_;
};

class Dsv41Chat : public ChatModel {
public:
    Dsv41Chat(const std::string & path, int ctx) : text_(path), max_ctx_(ctx), eng_((setenv("BL_DSV41_HYBRID", "1", 0), path), ctx) {
        // (hybrid decode unless the environment says otherwise: the exact fp32 path is for comparisons)
        const auto ids = text_.tokenize("</think>");
        if (ids.size() == 1) think_end_ = ids[0];
        // the expert tiers warmed from the usage profile: BL_DSV41_PROFILE, else bench/dsv41/expert_profile.txt next to
        // the build directory of this executable ("none": cold start)
        std::string prof = std::getenv("BL_DSV41_PROFILE") ? std::getenv("BL_DSV41_PROFILE") : "";
        if (prof.empty()) {
            char exe[4096];
            const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
            if (n > 0) {
                std::string d(exe, static_cast<size_t>(n));
                d = d.substr(0, d.rfind('/'));
                prof = d.substr(0, d.rfind('/')) + "/bench/dsv41/expert_profile.txt";
            }
        }
        if (prof != "none") {
            const auto t0 = std::chrono::steady_clock::now();
            const int placed = eng_.warm(prof);
            if (placed > 0)
                std::fprintf(stderr, "dsv41: expert tiers warmed in %.0f s (%d in VRAM) from %s\n",
                             std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), placed, prof.c_str());
        }
    }
    void        reset() override { eng_.reset(); session_.clear(); pending_ = -1; }
    int         position() const override { return eng_.position() + (pending_ >= 0 ? 1 : 0); }
    int         max_ctx() const override { return max_ctx_; }
    std::string report() const override { return eng_.report(); }
    std::string family() const override { return "deepseek-v4.1-flash"; }
    std::string tool_call_tag() const override { return "<｜DSML｜ calls>"; }

    ChatResult reply(const std::vector<Text::Message> & messages, const ChatParams & p,
                     const std::function<bool(const std::string &, bool)> & on_text) override {
        using clock = std::chrono::steady_clock;
        ChatResult res;
        const bool think = !p.think.empty();
        const std::string rendered = render_dsv41(messages, p.think, p.tools);
        // the session's text a prefix of the new conversation: feed the rest (after the pending token); else start over
        std::string suffix;
        if (!session_.empty() && rendered.size() > session_.size() && rendered.compare(0, session_.size(), session_) == 0) {
            suffix = rendered.substr(session_.size());
            res.n_reused = position();
        } else {
            reset();
            suffix = rendered;
        }
        std::vector<int> toks;
        if (pending_ >= 0) toks.push_back(pending_);
        const std::vector<int> more = text_.tokenize(suffix);
        toks.insert(toks.end(), more.begin(), more.end());
        if (eng_.position() + static_cast<int>(toks.size()) + 1 > max_ctx_) {
            if (res.n_reused == 0) throw std::runtime_error("the conversation is longer than the context");
            reset();
            res.n_reused = 0;
            toks = text_.tokenize(rendered);
            if (static_cast<int>(toks.size()) + 1 > max_ctx_) throw std::runtime_error("the conversation is longer than the context");
        }
        pending_ = -1;
        res.n_prompt = static_cast<int>(toks.size());
        const int max_new = std::max(1, std::min(p.max_new, max_ctx_ - eng_.position() - res.n_prompt));
        if (p.sampling.seed) rng_.seed(p.sampling.seed);

        const auto t0 = clock::now();
        const std::vector<float> * logits = &eng_.prefill(toks);
        const auto t1 = clock::now();
        bool reasoning = think, stop = false;
        std::string generated;
        Utf8Stream utf8;
        while (true) {
            const int tok = sample(*logits, p.sampling);
            ++res.n_generated;
            generated += text_.piece(tok);
            const bool eog = text_.is_eog(tok);
            if (eog) res.stopped_eog = true;
            else if (tok == think_end_ && reasoning) {   // the reasoning is over; the answer follows
                reasoning = false;
                const std::string rest = utf8.flush();
                if (!rest.empty()) { res.reasoning += rest; if (on_text && !on_text(rest, true)) stop = true; }
            } else {
                std::string s = utf8.push(text_.piece(tok));
                if (!reasoning && res.content.empty()) {   // the blank lines after </think>
                    const auto b = s.find_first_not_of("\n");
                    s = b == std::string::npos ? "" : s.substr(b);
                }
                if (!s.empty()) {
                    (reasoning ? res.reasoning : res.content) += s;
                    if (on_text && !on_text(s, reasoning)) stop = true;
                }
            }
            if (eog || stop || res.n_generated >= max_new) { pending_ = tok; break; }   // the last token is not fed yet
            logits = &eng_.step(tok);
        }
        const std::string rest = utf8.flush();
        if (!rest.empty()) { (reasoning ? res.reasoning : res.content) += rest; if (on_text) on_text(rest, reasoning); }
        res.prompt_s = std::chrono::duration<double>(t1 - t0).count();
        res.gen_s = std::chrono::duration<double>(clock::now() - t1).count();
        session_ = rendered + generated;
        return res;
    }

private:
    // greedy at temperature 0; else top-k (at most 256), temperature, top-p, then a draw
    int sample(const std::vector<float> & l, const Sampling & s) {
        const int V = static_cast<int>(l.size());
        if (s.temperature <= 0.f) return static_cast<int>(std::max_element(l.begin(), l.end()) - l.begin());
        const int k = std::clamp(s.top_k > 0 ? s.top_k : 256, 1, std::min(256, V));
        std::vector<int> id(V);
        std::iota(id.begin(), id.end(), 0);
        std::partial_sort(id.begin(), id.begin() + k, id.end(), [&](int a, int b) { return l[a] > l[b]; });
        std::vector<double> pr(k);
        double z = 0;
        for (int i = 0; i < k; ++i) z += (pr[i] = std::exp((l[id[i]] - l[id[0]]) / s.temperature));
        int n = k;
        if (s.top_p < 1.f) {
            double c = 0;
            for (int i = 0; i < k; ++i) { c += pr[i] / z; if (c >= s.top_p) { n = i + 1; break; } }
        }
        double tot = 0;
        for (int i = 0; i < n; ++i) tot += pr[i];
        double r = std::uniform_real_distribution<double>(0, tot)(rng_);
        for (int i = 0; i < n; ++i) { r -= pr[i]; if (r <= 0) return id[i]; }
        return id[n - 1];
    }

    Text        text_;
    int         max_ctx_;
    Dsv41Engine eng_;
    int         think_end_ = -1;
    int         pending_ = -1;   // sampled, not fed: the first token of the next turn's input
    std::string session_;        // the text the session holds (fed + generated)
    std::mt19937_64 rng_{0x5eed};
};

}  // namespace

namespace {

// JSON the way Python's json.dumps writes it (", " and ": ", the keys in their order), as the official encoding does
std::string py_dump(const nlohmann::ordered_json & v) {
    if (v.is_object()) {
        std::string s = "{";
        for (auto it = v.begin(); it != v.end(); ++it) s += (it == v.begin() ? "" : ", ") + nlohmann::ordered_json(it.key()).dump() + ": " + py_dump(it.value());
        return s + "}";
    }
    if (v.is_array()) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + py_dump(v[i]);
        return s + "]";
    }
    return v.dump();
}

const char * kDsv41ToolPrompt = R"(## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<｜DSML｜ calls>" block like the following:

<｜DSML｜ calls>
<｜DSML｜ invoke name="$TOOL_NAME">
<｜DSML｜ parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</｜DSML｜ parameter>
...
</｜DSML｜ invoke>
<｜DSML｜ invoke name="$TOOL_NAME2">
...
</｜DSML｜ invoke>
</｜DSML｜ calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.

Otherwise, output directly after </think> with tool calls or final response.

### Available Tool Schemas

)";

// an assistant turn's calls as a DSML block (a string argument as is, any other value as JSON)
std::string dsv41_calls(const std::vector<Text::ToolCall> & calls) {
    std::string s = "\n\n<｜DSML｜ calls>\n";
    for (size_t i = 0; i < calls.size(); ++i) {
        const Text::ToolCall & c = calls[i];
        s += (i ? "\n" : "") + std::string("<｜DSML｜ invoke name=\"") + c.name + "\">\n";
        nlohmann::ordered_json args = nlohmann::ordered_json::object();
        if (!c.arguments.empty()) {
            try { args = nlohmann::ordered_json::parse(c.arguments); } catch (...) { args = {{"arguments", c.arguments}}; }
            if (!args.is_object()) args = {{"arguments", c.arguments}};
        } else {
            for (const auto & [k, v] : c.args) args[k] = v;   // no client JSON: every value as written, a string
        }
        size_t j = 0;
        for (auto it = args.begin(); it != args.end(); ++it, ++j) {
            const bool str = it.value().is_string();
            s += (j ? "\n" : "") + std::string("<｜DSML｜ parameter name=\"") + it.key() + "\" string=\"" + (str ? "true" : "false") + "\">" +
                 (str ? it.value().get<std::string>() : py_dump(it.value())) + "</｜DSML｜ parameter>";
        }
        s += "\n</｜DSML｜ invoke>";
    }
    return s + "\n</｜DSML｜ calls>";
}

}  // namespace

// deepseek-recipe's DeepseekV41Encoding (the official API's prompts; tests/test_dsv41_template.cpp holds its renderings):
// BOS; a first message preceded by <｜System｜> when it is a system message or when thinking, and when thinking by
// "Reasoning Effort: N ...\n\n" (low 50, max 100, else 75); system <｜System｜>text, user <｜User｜>text and tool
// <｜User｜><tool_result>text</tool_result> (after a user or tool turn: "\n\n" instead of <｜User｜>); an assistant turn
// <｜Assistant｜>, then </think> - or, thinking with tools, <think>its reasoning</think> - its text, its calls as a DSML
// block, <｜end▁of▁sentence｜>. With tools, the first system message (an empty one put first if there is none) gets
// "\n\n" and the tool prompt with one schema per line. The generation prompt: <｜Assistant｜><think> or </think>.
std::string render_dsv41(const std::vector<Text::Message> & in, const std::string & think_effort, const std::vector<std::string> & tools) {
    const bool think = !think_effort.empty();
    std::vector<Text::Message> msgs = in;
    for (auto & m : msgs)
        if (m.role == "developer") m.role = "system";
    if (!tools.empty()) {
        if (msgs.empty() || msgs[0].role != "system") msgs.insert(msgs.begin(), Text::Message{"system", "", ""});
        std::string t = kDsv41ToolPrompt;
        for (size_t i = 0; i < tools.size(); ++i) {   // an OpenAI tool ({"type", "function": {...}}) or the function itself
            const auto j = nlohmann::ordered_json::parse(tools[i]);
            const auto & fn = j.contains("function") ? j["function"] : j;
            nlohmann::ordered_json d = {{"name", fn.at("name")}, {"description", fn.value("description", "")},
                                        {"parameters", fn.contains("parameters") ? fn["parameters"] : nlohmann::ordered_json()}};
            t += (i ? "\n" : "") + py_dump(d);
        }
        msgs[0].content += "\n\n" + t + "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";
    }
    std::string s = "<｜begin▁of▁sentence｜>";
    for (size_t i = 0; i < msgs.size(); ++i) {
        const Text::Message & m = msgs[i];
        const std::string & role = m.role;
        if (i == 0 && (think || role == "system")) s += "<｜System｜>";
        if (i == 0 && think)
            s += "Reasoning Effort: " + std::string(think_effort == "low" ? "50" : think_effort == "max" ? "100" : "75") +
                 " (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
        const bool after_user = i > 0 && (msgs[i - 1].role == "user" || msgs[i - 1].role == "tool");
        if (role == "system") s += (i > 0 ? "<｜System｜>" : "") + m.content;
        else if (role == "user") s += (after_user ? "\n\n" : "<｜User｜>") + m.content;
        else if (role == "tool") s += (after_user ? "\n\n" : "<｜User｜>") + ("<tool_result>" + m.content + "</tool_result>");
        else if (role == "assistant") {
            // earlier reasoning only with tools (the official encoding drops it otherwise)
            const bool reasoning = think && i > 0 && !tools.empty();
            s += reasoning ? "<｜Assistant｜><think>" + m.reasoning + "</think>" : "<｜Assistant｜></think>";
            s += m.content;
            if (!m.tool_calls.empty()) s += dsv41_calls(m.tool_calls);
            s += "<｜end▁of▁sentence｜>";
        } else {
            throw std::runtime_error("this model's template takes system, user, assistant and tool messages (got " + m.role + ")");
        }
    }
    return s + (think ? "<｜Assistant｜><think>" : "<｜Assistant｜></think>");
}

std::unique_ptr<ChatModel> open_chat(const std::string & shard1, int max_ctx, const std::string & cache_file) {
    if (architecture(shard1) == "deepseek41") return std::make_unique<Dsv41Chat>(shard1, max_ctx);
    return std::make_unique<QwenChat>(shard1, max_ctx, cache_file);
}

}  // namespace bl
