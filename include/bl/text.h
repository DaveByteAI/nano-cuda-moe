// Text in and out: the model's tokenizer (llama.cpp's, loaded vocab-only from the GGUF: no weights), the chat
// template (the model's Jinja template for text messages, written out), and UTF-8-safe streaming of token pieces.
#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace bl {

class Text {
public:
    explicit Text(const std::string & gguf_path);
    ~Text();
    Text(const Text &) = delete;
    Text & operator=(const Text &) = delete;

    std::vector<int> tokenize(const std::string & s, bool parse_special = true) const;
    std::string      piece(int token) const;   // its bytes (special tokens rendered)
    std::string      detokenize(const std::vector<int> & tokens) const;
    bool             is_eog(int token) const;  // end of generation (<|im_end|>, <|endoftext|>, ...)

    struct ToolCall {
        std::string name;
        std::vector<std::pair<std::string, std::string>> args;   // name, value as written (a string as is, else JSON)
        std::string arguments = {};   // the arguments object as the client sent it (JSON; deepseek41 writes them from it)
    };
    struct Message {
        std::string role;        // system | developer | user | assistant | tool
        std::string content;
        std::string reasoning;   // an assistant turn's thinking (optional)
        std::vector<ToolCall> tool_calls = {};   // an assistant turn's function calls
    };
    // the reasoning effort when thinking: "" = off, "low", "medium", "xhigh"; tools: the function definitions, each one
    // JSON as the template's tojson writes it
    static std::string render(const std::vector<Message> & messages, bool add_generation_prompt, const std::string & think,
                              const std::vector<std::string> & tools = {});

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

// bytes of token pieces in, complete UTF-8 characters out (a character can span pieces)
class Utf8Stream {
public:
    std::string push(const std::string & bytes);
    std::string flush();   // whatever is left, as is
private:
    std::string pend_;
};

}  // namespace bl
