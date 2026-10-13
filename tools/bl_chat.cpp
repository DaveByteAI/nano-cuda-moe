// Interactive chat in the terminal.
//   bl-chat --model SHARD1.gguf [--ctx 32768] [--temp 0.7] [--top-p 0.8] [--top-k 20] [--seed N] [--spec 3]
//           [--think off|low|medium|xhigh] [--system TEXT] [--max-new 4096]
// Commands: /reset (new conversation), /think MODE, /temp X, /stats, /exit
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "bl/chat.h"

int main(int argc, char ** argv) {
    std::string model, system, think;
    int ctx = 32768;
    bl::ChatParams cp;
    cp.sampling.temperature = 0.7f;
    cp.sampling.top_p = 0.8f;
    cp.sampling.top_k = 20;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : ""; };
        if (a == "--model") model = next();
        else if (a == "--ctx") ctx = bl::parse_ctx(next());
        else if (a == "--temp") cp.sampling.temperature = std::stof(next());
        else if (a == "--top-p") cp.sampling.top_p = std::stof(next());
        else if (a == "--top-k") cp.sampling.top_k = std::stoi(next());
        else if (a == "--seed") cp.sampling.seed = std::stoull(next());
        else if (a == "--spec") cp.spec = std::stoi(next());
        else if (a == "--max-new") cp.max_new = std::stoi(next());
        else if (a == "--think") { think = next(); cp.think = think == "off" ? "" : think; }
        else if (a == "--system") system = next();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (model.empty()) { std::fprintf(stderr, "usage: %s --model SHARD1.gguf [options]\n", argv[0]); return 2; }
    try {
        std::fprintf(stderr, "loading %s ...\n", model.c_str());
        const auto t0 = std::chrono::steady_clock::now();
        bl::Text text(model);
        bl::Engine eng(model, ctx);
        bl::Chat chat(eng, text, ctx);
        std::fprintf(stderr, "ready in %.0f s. /reset starts over, /think off|low|medium|xhigh, /temp X, /stats, /exit\n",
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        std::vector<bl::Text::Message> msgs;
        if (!system.empty()) msgs.push_back({"system", system, ""});
        bl::ChatResult last;
        std::string line;
        while (true) {
            std::printf("\n\033[1m> \033[0m");
            std::fflush(stdout);
            if (!std::getline(std::cin, line)) break;
            if (line.empty()) continue;
            if (line == "/exit" || line == "/quit") break;
            if (line == "/reset") { chat.reset(); msgs.clear(); if (!system.empty()) msgs.push_back({"system", system, ""}); std::printf("(new conversation)\n"); continue; }
            if (line.rfind("/think", 0) == 0) { const std::string m = line.size() > 7 ? line.substr(7) : "off"; cp.think = m == "off" ? "" : m; std::printf("(thinking: %s)\n", m.c_str()); continue; }
            if (line.rfind("/temp", 0) == 0) { cp.sampling.temperature = std::stof(line.substr(5)); std::printf("(temperature %.2f)\n", cp.sampling.temperature); continue; }
            if (line == "/stats") {
                std::printf("context %d / %d tokens\n%s\n", chat.position(), ctx, eng.report().c_str());
                continue;
            }
            msgs.push_back({"user", line, ""});
            bool in_reasoning = false;
            try {
                last = chat.reply(msgs, cp, [&](const std::string & s, bool reasoning) {
                    if (reasoning != in_reasoning) { std::printf(reasoning ? "\033[2m" : "\033[0m\n"); in_reasoning = reasoning; }
                    std::fwrite(s.data(), 1, s.size(), stdout);
                    std::fflush(stdout);
                    return true;
                });
            } catch (const std::exception & e) {
                std::printf("\033[0m\n(error: %s)\n", e.what());
                msgs.pop_back();
                continue;
            }
            if (in_reasoning) std::printf("\033[0m");
            msgs.push_back({"assistant", last.content, last.reasoning});
            std::printf("\n\033[2m[%d prompt tokens%s, %.2f s; %d tokens in %.2f s, %.1f tok/s%s]\033[0m\n", last.n_prompt,
                        last.n_reused ? (" (+" + std::to_string(last.n_reused) + " reused)").c_str() : "", last.prompt_s,
                        last.n_generated, last.gen_s, last.gen_s > 0 ? (last.n_generated - 1) / last.gen_s : 0.0,
                        last.stopped_eog ? "" : ", cut at max-new");
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
