#include "bl/text.h"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "gguf.h"
#include "llama.h"

namespace bl {

struct Text::Impl {
    llama_model * model = nullptr;
    const llama_vocab * vocab = nullptr;
};

namespace {

// a model whose architecture this libllama does not know (deepseek41) still has a tokenizer it knows (the vocab, the
// merges, a pre-tokenizer such as joyai-llm): its metadata, tensors left out, copied to a temporary file under an
// architecture name libllama accepts. A vocab-only load reads nothing architecture-specific. "" if not applicable.
std::string vocab_copy(const std::string & path) {
    gguf_init_params ip{true, nullptr};
    gguf_context * src = gguf_init_from_file(path.c_str(), ip);
    if (!src) return "";
    std::string out;
    const int64_t k = gguf_find_key(src, "general.architecture");
    if (k >= 0 && gguf_find_key(src, "tokenizer.ggml.model") >= 0) {
        gguf_context * dst = gguf_init_empty();
        gguf_set_kv(dst, src);
        gguf_set_val_str(dst, "general.architecture", "deepseek2");
        for (const char * key : {"split.no", "split.count", "split.tensors.count"}) gguf_remove_key(dst, key);   // one file
        char tmpl[] = "/tmp/bl-vocab-XXXXXX";
        const int fd = mkstemp(tmpl);
        if (fd >= 0) {
            ::close(fd);
            if (gguf_write_to_file(dst, tmpl, true)) out = tmpl;
            else std::remove(tmpl);
        }
        gguf_free(dst);
    }
    gguf_free(src);
    return out;
}

}  // namespace

Text::Text(const std::string & path) : p_(std::make_unique<Impl>()) {
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);   // quiet
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    mp.n_gpu_layers = 0;
    p_->model = llama_model_load_from_file(path.c_str(), mp);
    if (!p_->model) {   // an architecture unknown here: its tokenizer through a vocab-only copy
        const std::string tmp = vocab_copy(path);
        if (!tmp.empty()) {
            p_->model = llama_model_load_from_file(tmp.c_str(), mp);
            std::remove(tmp.c_str());
        }
    }
    if (!p_->model) throw std::runtime_error("cannot load the tokenizer from " + path);
    p_->vocab = llama_model_get_vocab(p_->model);
}

Text::~Text() {
    if (p_ && p_->model) llama_model_free(p_->model);
}

std::vector<int> Text::tokenize(const std::string & s, bool parse_special) const {
    std::vector<llama_token> t(s.size() + 8);
    int n = llama_tokenize(p_->vocab, s.data(), static_cast<int32_t>(s.size()), t.data(), static_cast<int32_t>(t.size()), false,
                           parse_special);
    if (n < 0) {
        t.resize(static_cast<size_t>(-n));
        n = llama_tokenize(p_->vocab, s.data(), static_cast<int32_t>(s.size()), t.data(), static_cast<int32_t>(t.size()), false,
                           parse_special);
    }
    if (n < 0) throw std::runtime_error("tokenize failed");
    return std::vector<int>(t.begin(), t.begin() + n);
}

std::string Text::piece(int token) const {
    char buf[256];
    int n = llama_token_to_piece(p_->vocab, token, buf, sizeof buf, 0, true);
    if (n < 0) {
        std::string big(static_cast<size_t>(-n), '\0');
        n = llama_token_to_piece(p_->vocab, token, big.data(), static_cast<int32_t>(big.size()), 0, true);
        return big.substr(0, n > 0 ? static_cast<size_t>(n) : 0);
    }
    return std::string(buf, static_cast<size_t>(n));
}

std::string Text::detokenize(const std::vector<int> & tokens) const {
    std::string s;
    for (int t : tokens) s += piece(t);
    return s;
}

bool Text::is_eog(int token) const { return llama_vocab_is_eog(p_->vocab, token); }

namespace {
std::string trim(const std::string & s) {
    const auto b = s.find_first_not_of(" \t\r\n"), e = s.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? "" : s.substr(b, e - b + 1);
}
}  // namespace

// the model's template (bench/prompts/chat_template.jinja) for text messages, with function calling
std::string Text::render(const std::vector<Message> & msgs, bool add_generation_prompt, const std::string & think,
                         const std::vector<std::string> & tools) {
    std::string out, system;
    size_t i = 0;
    for (; i < msgs.size() && (msgs[i].role == "system" || msgs[i].role == "developer"); ++i) {
        const std::string c = trim(msgs[i].content);
        if (!c.empty()) system += (system.empty() ? "" : "\n") + c;
    }
    std::string effort;
    if (think == "xhigh" || think == "high")
        effort = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider "
                 "plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    else if (think == "low")
        effort = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without "
                 "unnecessary elaboration.";
    if (!tools.empty()) {   // the function definitions and how to call them, then the system text
        out += "<|im_start|>system\n" + (effort.empty() ? "" : effort + "\n\n") +
               "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const std::string & t : tools) out += "\n" + t;
        out += "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
               "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
               "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
               "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
               "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
               "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
               "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
               "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";
        if (!system.empty()) out += "\n\n" + system;
        out += "<|im_end|>\n";
    } else if (!system.empty()) out += "<|im_start|>system\n" + (effort.empty() ? "" : effort + "\n\n") + system + "<|im_end|>\n";
    else if (!effort.empty()) out += "<|im_start|>system\n" + effort + "<|im_end|>\n";
    for (; i < msgs.size(); ++i) {
        const Message & m = msgs[i];
        const std::string c = trim(m.content);
        if (m.role == "system" || m.role == "developer") throw std::runtime_error("a system message must come first");
        if (m.role == "user") out += "<|im_start|>user\n" + c + "<|im_end|>\n";
        else if (m.role == "assistant") {
            out += "<|im_start|>assistant\n<think>\n" + trim(m.reasoning) + "\n</think>\n\n" + c;
            for (size_t k = 0; k < m.tool_calls.size(); ++k) {
                const ToolCall & tc = m.tool_calls[k];
                out += (k ? "\n<tool_call>\n<function=" : c.empty() ? "<tool_call>\n<function=" : "\n\n<tool_call>\n<function=") + tc.name + ">\n";
                for (const auto & [name, value] : tc.args) out += "<parameter=" + name + ">\n" + value + "\n</parameter>\n";
                out += "</function>\n</tool_call>";
            }
            out += "<|im_end|>\n";
        }
        else if (m.role == "tool") {
            if (i == 0 || msgs[i - 1].role != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + c + "\n</tool_response>";
            if (i + 1 == msgs.size() || msgs[i + 1].role != "tool") out += "<|im_end|>\n";
        } else throw std::runtime_error("unexpected role " + m.role);
    }
    if (add_generation_prompt) out += think.empty() ? "<|im_start|>assistant\n<think>\n\n</think>\n\n" : "<|im_start|>assistant\n<think>\n";
    return out;
}

std::string Utf8Stream::push(const std::string & bytes) {
    pend_ += bytes;
    // the longest prefix that ends on a character boundary
    size_t end = pend_.size(), i = pend_.size();
    int back = 0;
    while (i > 0 && back < 4) {
        const unsigned char ch = static_cast<unsigned char>(pend_[i - 1]);
        if ((ch & 0xC0) != 0x80) {   // a lead (or ASCII) byte at i-1: is its character complete?
            const int len = ch < 0x80 ? 1 : (ch >> 5) == 6 ? 2 : (ch >> 4) == 14 ? 3 : (ch >> 3) == 30 ? 4 : 1;
            end = (pend_.size() - (i - 1) >= static_cast<size_t>(len)) ? pend_.size() : i - 1;
            break;
        }
        --i;
        ++back;
    }
    std::string out = pend_.substr(0, end);
    pend_.erase(0, end);
    return out;
}

std::string Utf8Stream::flush() {
    std::string out;
    out.swap(pend_);
    return out;
}

}  // namespace bl
