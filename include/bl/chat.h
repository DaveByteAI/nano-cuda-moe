// A conversation on top of the engine: messages -> the chat template -> tokens -> generate, streaming text out.
// The engine keeps the session (KV, GDN states): when the new conversation's text extends what the session already
// holds (the usual next turn), only the new part is fed; otherwise the session restarts from the whole conversation.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "bl/engine.h"
#include "bl/text.h"

namespace bl {

struct ChatParams {
    Sampling    sampling;            // temperature 0: greedy
    int         spec = 3;            // MTP drafts per round (0: none)
    int         max_new = 4096;
    std::string think;               // "" = no thinking; "low" / "medium" / "xhigh"
    std::vector<std::string> tools;  // function definitions (JSON each, as the template's tojson writes it)
};

struct ChatResult {
    std::string content, reasoning;
    int    n_generated = 0;          // tokens, the end-of-generation one included
    int    n_prompt = 0, n_reused = 0;   // prompt tokens fed / already in the session
    bool   stopped_eog = false;      // ended by the model (else max_new or the context)
    double prompt_s = 0, gen_s = 0;
};

class Chat {
public:
    Chat(Engine & e, const Text & t, int max_ctx);
    // on_text(piece, is_reasoning) gets the text as it is generated; returning false stops the generation
    ChatResult reply(const std::vector<Text::Message> & messages, const ChatParams & p,
                     const std::function<bool(const std::string &, bool)> & on_text = nullptr);
    void reset();
    int  position() const { return eng_.position(); }

private:
    Engine &     eng_;
    const Text & txt_;
    int          max_ctx_;
    int          think_end_ = -1;    // the </think> token
    std::string  session_;           // the text the session holds (fed + generated)
};

// A chat model as the programs see it, whichever engine runs it: open_chat picks by the GGUF's architecture
// (qwen3.8's Engine + Chat, or the deepseek41 engine with its own template and host-side sampling)
class ChatModel {
public:
    virtual ~ChatModel() = default;
    virtual ChatResult reply(const std::vector<Text::Message> & messages, const ChatParams & p,
                             const std::function<bool(const std::string &, bool)> & on_text = nullptr) = 0;
    virtual void        reset() = 0;
    virtual int         position() const = 0;
    virtual std::string report() const = 0;
    virtual std::string family() const = 0;           // a model name prefix: "qwen3.8-flash", "deepseek-v4.1-flash"
    virtual bool        tools() const { return true; } // function calling through the template
    virtual void        save_cache(const std::string &) {}
};
// cache_file: the qwen3.8 engine's --expert-cache (ignored by the others)
std::unique_ptr<ChatModel> open_chat(const std::string & shard1, int max_ctx, const std::string & cache_file = "");

}  // namespace bl
