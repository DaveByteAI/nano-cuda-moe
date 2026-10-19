// The ChatModel implementations: qwen3.8 (Engine + Chat as they are) and deepseek41 (Dsv41Engine; the template from
// the GGUF's tokenizer.chat_template written out, the session kept between turns, sampling on the host).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>

#include "bl/chat.h"
#include "bl/dsv41.h"
#include "gguf.h"

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
    QwenChat(const std::string & path, int ctx, const std::string & cache) : text_(path), eng_(path, ctx, cache), chat_(eng_, text_, ctx) {}
    ChatResult reply(const std::vector<Text::Message> & m, const ChatParams & p,
                     const std::function<bool(const std::string &, bool)> & on_text) override { return chat_.reply(m, p, on_text); }
    void        reset() override { chat_.reset(); }
    int         position() const override { return chat_.position(); }
    std::string report() const override { return eng_.report(); }
    std::string family() const override { return "qwen3.8-flash"; }
    void        save_cache(const std::string & f) override { eng_.save_cache(f); }
private:
    Text   text_;
    Engine eng_;
    Chat   chat_;
};

// tokenizer.chat_template of the deepseek41 GGUF ("checked against the exact release's encoding.py"): BOS, system as
// <｜System｜>, a user turn as <｜User｜> (consecutive ones joined by a blank line), an assistant turn as its text and
// <｜end▁of▁sentence｜>; after a user turn (or a later system one) that the assistant answers: <｜Assistant｜></think>.
// Thinking (not in that template, the release's thinking mode): the generation prompt ends <｜Assistant｜><think> instead.
std::string render_dsv41(const std::vector<Text::Message> & msgs, bool think) {
    std::string s = "<｜begin▁of▁sentence｜>";
    for (size_t i = 0; i < msgs.size(); ++i) {
        const Text::Message & m = msgs[i];
        const std::string role = m.role == "developer" ? "system" : m.role;
        if (!m.tool_calls.empty()) throw std::runtime_error("this model's template has no tool calls");
        if (role == "system") s += "<｜System｜>" + m.content;
        else if (role == "user") s += (i > 0 && msgs[i - 1].role == "user" ? "\n\n" : "<｜User｜>") + m.content;
        else if (role == "assistant") s += m.content + "<｜end▁of▁sentence｜>";
        else throw std::runtime_error("this model's template takes system, user and assistant messages (got " + m.role + ")");
        const bool last = i + 1 == msgs.size();
        if ((last || msgs[i + 1].role == "assistant") && (role == "user" || (role == "system" && i > 0)))
            s += last && think ? "<｜Assistant｜><think>" : "<｜Assistant｜></think>";
    }
    return s;
}

class Dsv41Chat : public ChatModel {
public:
    Dsv41Chat(const std::string & path, int ctx) : text_(path), max_ctx_(ctx), eng_((setenv("BL_DSV41_HYBRID", "1", 0), path), ctx) {
        // (hybrid decode unless the environment says otherwise: the exact fp32 path is for comparisons)
        const auto ids = text_.tokenize("</think>");
        if (ids.size() == 1) think_end_ = ids[0];
    }
    void        reset() override { eng_.reset(); session_.clear(); pending_ = -1; }
    int         position() const override { return eng_.position() + (pending_ >= 0 ? 1 : 0); }
    std::string report() const override { return eng_.report(); }
    std::string family() const override { return "deepseek-v4.1-flash"; }
    bool        tools() const override { return false; }

    ChatResult reply(const std::vector<Text::Message> & messages, const ChatParams & p,
                     const std::function<bool(const std::string &, bool)> & on_text) override {
        using clock = std::chrono::steady_clock;
        if (!p.tools.empty()) throw std::runtime_error("this model's template has no tool calls");
        ChatResult res;
        const bool think = !p.think.empty();
        const std::string rendered = render_dsv41(messages, think);
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

std::unique_ptr<ChatModel> open_chat(const std::string & shard1, int max_ctx, const std::string & cache_file) {
    if (architecture(shard1) == "deepseek41") return std::make_unique<Dsv41Chat>(shard1, max_ctx);
    return std::make_unique<QwenChat>(shard1, max_ctx, cache_file);
}

}  // namespace bl
