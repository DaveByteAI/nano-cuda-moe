#include "bl/chat.h"

#include <chrono>
#include <stdexcept>

namespace bl {

Chat::Chat(Engine & e, const Text & t, int max_ctx) : eng_(e), txt_(t), max_ctx_(max_ctx) {
    const auto ids = txt_.tokenize("</think>");
    if (ids.size() == 1) think_end_ = ids[0];
}

void Chat::reset() {
    eng_.reset();
    session_.clear();
}

ChatResult Chat::reply(const std::vector<Text::Message> & messages, const ChatParams & p,
                       const std::function<bool(const std::string &, bool)> & on_text) {
    using clock = std::chrono::steady_clock;
    ChatResult res;
    const std::string rendered = Text::render(messages, true, p.think, p.tools);
    // the session's text a prefix of the new conversation: feed the rest; else start over
    std::string suffix;
    if (!session_.empty() && rendered.compare(0, session_.size(), session_) == 0 && rendered.size() > session_.size()) {
        suffix = rendered.substr(session_.size());
        res.n_reused = eng_.position() + (eng_.pending() >= 0 ? 1 : 0);
    } else {
        reset();
        suffix = rendered;
    }
    std::vector<int> toks = txt_.tokenize(suffix);
    const int used = eng_.position() + (eng_.pending() >= 0 ? 1 : 0);
    if (used + static_cast<int>(toks.size()) + 1 > max_ctx_) {   // does not fit after the session: from scratch, if that fits
        if (res.n_reused == 0) throw std::runtime_error("the conversation is longer than the context");
        reset();
        res.n_reused = 0;
        toks = txt_.tokenize(rendered);
        if (static_cast<int>(toks.size()) + 1 > max_ctx_) throw std::runtime_error("the conversation is longer than the context");
    }
    res.n_prompt = static_cast<int>(toks.size());
    const int room = max_ctx_ - (eng_.position() + (eng_.pending() >= 0 ? 1 : 0)) - res.n_prompt;
    const int max_new = std::max(1, std::min(p.max_new, room));

    bool reasoning = !p.think.empty();   // a thinking turn starts inside <think>
    bool stop = false;
    std::string generated;
    Utf8Stream utf8;
    const auto t0 = clock::now();
    auto t1 = t0;
    eng_.generate(toks, max_new, p.spec, 0.5f, [&](int tok) {
        if (res.n_generated++ == 0) t1 = clock::now();
        generated += txt_.piece(tok);
        if (txt_.is_eog(tok)) { res.stopped_eog = true; return false; }
        if (tok == think_end_ && reasoning) {   // the reasoning is over; the answer follows (after "\n\n")
            reasoning = false;
            const std::string rest = utf8.flush();
            if (!rest.empty()) { res.reasoning += rest; if (on_text && !on_text(rest, true)) stop = true; }
            return !stop;
        }
        std::string s = utf8.push(txt_.piece(tok));
        if (s.empty()) return true;
        if (reasoning) res.reasoning += s;
        else {
            if (res.content.empty()) {   // the blank lines after </think>
                const auto b = s.find_first_not_of("\n");
                s = b == std::string::npos ? "" : s.substr(b);
                if (s.empty()) return true;
            }
            res.content += s;
        }
        if (on_text && !on_text(s, reasoning)) stop = true;
        return !stop;
    }, &p.sampling);
    const std::string rest = utf8.flush();
    if (!rest.empty()) { (reasoning ? res.reasoning : res.content) += rest; if (on_text) on_text(rest, reasoning); }
    const auto t2 = clock::now();
    res.prompt_s = std::chrono::duration<double>(t1 - t0).count();
    res.gen_s = std::chrono::duration<double>(t2 - t1).count();
    session_ = rendered + generated;   // what the session now holds (the last token pending)
    return res;
}

}  // namespace bl
