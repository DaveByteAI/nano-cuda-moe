// bl-dspark-pack: DeepSeek-V4.1-Flash's DSpark draft (the official checkpoint's mtp.0-2 tensors, shards 44-46 of 48)
// -> one GGUF next to the main model's.
//   bl-dspark-pack SRC_DIR OUT.gguf      (SRC_DIR: config.json and the three model-000NN-of-00048.safetensors)
// The routed experts (MXFP4: e2m1 pairs, low nibble first, an E8M0 scale per 32 values) become IQ4_XS (4.25 bits, as
// MXFP4); the FP8 (E4M3) matrices with their 32x32 E8M0 block scales become Q8_0; the router and the Markov head stay
// BF16; vectors become F32. Names follow the main model's GGUF ("dspark.<stage>.attn_q_a.weight", ...), experts
// stacked [in][rows][expert] like ffn_*_exps.
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

namespace {

struct St {   // one safetensors tensor
    std::string dtype;
    std::vector<int64_t> shape;
    const uint8_t * data = nullptr;
    size_t bytes = 0;
};

float bf16(uint16_t v) {
    const uint32_t u = static_cast<uint32_t>(v) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
float e8m0(uint8_t e) { return std::ldexp(1.0f, static_cast<int>(e) - 127); }
float e4m3(uint8_t b) {
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    const float v = e == 0 ? std::ldexp(m / 8.0f, -6) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return s ? -v : v;
}
const float kFp4[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// row o of a weight as floats: FP8 with 32x32 block scales, MXFP4 with per-32 scales, BF16 or F32
void row_floats(const St & w, const St * scale, int64_t o, float * out) {
    const int64_t in = w.dtype == "I8" ? 2 * w.shape[1] : w.shape[1];
    if (w.dtype == "F8_E4M3") {
        const uint8_t * r = w.data + o * in;
        const uint8_t * s = scale->data + (o / 32) * scale->shape[1];
        for (int64_t i = 0; i < in; ++i) out[i] = e4m3(r[i]) * e8m0(s[i / 32]);
    } else if (w.dtype == "I8") {
        const uint8_t * r = w.data + o * (in / 2);
        const uint8_t * s = scale->data + o * scale->shape[1];
        for (int64_t i = 0; i < in; i += 2) {
            const float d = e8m0(s[i / 32]);
            out[i] = kFp4[r[i / 2] & 15] * d;
            out[i + 1] = kFp4[r[i / 2] >> 4] * d;
        }
    } else if (w.dtype == "BF16") {
        const uint16_t * r = reinterpret_cast<const uint16_t *>(w.data) + o * in;
        for (int64_t i = 0; i < in; ++i) out[i] = bf16(r[i]);
    } else if (w.dtype == "F32") {
        std::memcpy(out, reinterpret_cast<const float *>(w.data) + o * in, in * 4);
    } else {
        throw std::runtime_error("unexpected dtype " + w.dtype);
    }
}

// the main model's names for the checkpoint's (stage prefix removed)
const std::map<std::string, std::string> kNames = {
    {"attn.wq_a.weight", "attn_q_a.weight"},          {"attn.q_norm.weight", "attn_q_a_norm.weight"},
    {"attn.wq_b.weight", "attn_q_b.weight"},          {"attn.wkv.weight", "attn_kv.weight"},
    {"attn.kv_norm.weight", "attn_kv_a_norm.weight"}, {"attn.wo_a.weight", "attn_output_a.weight"},
    {"attn.wo_b.weight", "attn_output_b.weight"},     {"attn.attn_sink", "attn_sinks.weight"},
    {"attn_norm.weight", "attn_norm.weight"},         {"ffn_norm.weight", "ffn_norm.weight"},
    {"ffn.gate.weight", "ffn_gate_inp.weight"},       {"ffn.gate.bias", "exp_probs_b.bias"},
    {"ffn.shared_experts.w1.weight", "ffn_gate_shexp.weight"}, {"ffn.shared_experts.w3.weight", "ffn_up_shexp.weight"},
    {"ffn.shared_experts.w2.weight", "ffn_down_shexp.weight"},
    {"hc_attn_fn", "hc_attn_fn.weight"},   {"hc_attn_base", "hc_attn_base.weight"},   {"hc_attn_scale", "hc_attn_scale.weight"},
    {"hc_ffn_fn", "hc_ffn_fn.weight"},     {"hc_ffn_base", "hc_ffn_base.weight"},     {"hc_ffn_scale", "hc_ffn_scale.weight"},
    {"main_proj.weight", "main_proj.weight"},           {"main_norm.weight", "main_norm.weight"},
    {"norm.weight", "output_norm.weight"},               {"markov_head.embed.weight", "markov_embd.weight"},
    {"markov_head.head.weight", "markov_head.weight"},   {"confidence_head.proj.weight", "confidence.weight"},
};

// rows [r0, r0 + n) of floats -> `type`, threaded over rows
std::vector<uint8_t> quantize(ggml_type type, int64_t rows, int64_t in, const std::function<void(int64_t, float *)> & get) {
    const size_t rb = ggml_row_size(type, in);
    std::vector<uint8_t> out(rb * rows);
    std::atomic<int64_t> next{0};
    std::vector<std::thread> pool;
    const int64_t chunk = 64;
    for (unsigned t = 0; t < std::max(1u, std::thread::hardware_concurrency()); ++t)
        pool.emplace_back([&] {
            std::vector<float> buf(static_cast<size_t>(chunk * in));
            for (int64_t r0; (r0 = next.fetch_add(chunk)) < rows;) {
                const int64_t n = std::min(chunk, rows - r0);
                for (int64_t r = 0; r < n; ++r) get(r0 + r, buf.data() + r * in);
                ggml_quantize_chunk(type, buf.data(), out.data() + rb * r0, 0, n, in, nullptr);
            }
        });
    for (auto & p : pool) p.join();
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s SRC_DIR OUT.gguf\n", argv[0]);
        return 2;
    }
    try {
        const std::string src = argv[1];
        std::ifstream cf(src + "/config.json");
        if (!cf) throw std::runtime_error(src + "/config.json: not found");
        nlohmann::json cfg = nlohmann::json::parse(cf);
        if (cfg.contains("text_config")) cfg = cfg["text_config"];

        std::map<std::string, St> ts;   // every mtp.* tensor, mapped
        for (int n : {44, 45, 46}) {
            const std::string path = src + "/model-000" + std::to_string(n) + "-of-00048.safetensors";
            const int fd = ::open(path.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error(path + ": cannot open");
            const off_t size = ::lseek(fd, 0, SEEK_END);
            void * m = ::mmap(nullptr, static_cast<size_t>(size), PROT_READ, MAP_PRIVATE, fd, 0);
            ::close(fd);
            if (m == MAP_FAILED) throw std::runtime_error(path + ": mmap failed");
            const uint8_t * base = static_cast<const uint8_t *>(m);
            uint64_t hl;
            std::memcpy(&hl, base, 8);
            const nlohmann::json h = nlohmann::json::parse(std::string(reinterpret_cast<const char *>(base + 8), hl));
            for (auto it = h.begin(); it != h.end(); ++it) {
                if (it.key() == "__metadata__") continue;
                St t;
                t.dtype = it.value()["dtype"];
                t.shape = it.value()["shape"].get<std::vector<int64_t>>();
                const auto off = it.value()["data_offsets"];
                t.data = base + 8 + hl + off[0].get<size_t>();
                t.bytes = off[1].get<size_t>() - off[0].get<size_t>();
                ts[it.key()] = t;
            }
        }
        ggml_quantize_init(GGML_TYPE_IQ4_XS);

        ggml_init_params ip{static_cast<size_t>(64) << 20, nullptr, true};
        ggml_context * ctx = ggml_init(ip);
        gguf_context * gg = gguf_init_empty();
        gguf_set_val_str(gg, "general.architecture", "deepseek41-dspark");
        gguf_set_val_str(gg, "dspark.source", "deepseek-ai/DeepSeek-V4.1-Flash-1017 mtp.* (shards 44-46 of 48)");
        const int n_stage = cfg.value("num_nextn_predict_layers", 0);
        gguf_set_val_u32(gg, "dspark.stage_count", static_cast<uint32_t>(n_stage));
        gguf_set_val_u32(gg, "dspark.block_size", cfg.value("dspark_block_size", 0));
        gguf_set_val_u32(gg, "dspark.noise_token_id", cfg.value("dspark_noise_token_id", 0));
        gguf_set_val_u32(gg, "dspark.markov_rank", cfg.value("dspark_markov_rank", 0));
        gguf_set_val_u32(gg, "dspark.expert_count", cfg.value("dspark_n_routed_experts", 0));
        gguf_set_val_u32(gg, "dspark.expert_used_count", cfg.value("dspark_num_experts_per_tok", 0));
        std::vector<int32_t> target;
        for (auto v : cfg["dspark_target_layer_ids"]) target.push_back(v.get<int32_t>());
        gguf_set_arr_data(gg, "dspark.target_layers", GGUF_TYPE_INT32, target.data(), target.size());

        std::vector<std::vector<uint8_t>> keep;
        auto add = [&](const std::string & name, ggml_type type, std::vector<int64_t> ne, std::vector<uint8_t> && bytes) {
            ggml_tensor * gt = ggml_new_tensor(ctx, type, static_cast<int>(ne.size()), ne.data());
            ggml_set_name(gt, name.c_str());
            keep.push_back(std::move(bytes));
            gt->data = keep.back().data();
            gguf_add_tensor(gg, gt);
            std::printf("%-44s %-7s", name.c_str(), ggml_type_name(type));
            for (int64_t v : ne) std::printf(" %lld", static_cast<long long>(v));
            std::printf("\n");
            std::fflush(stdout);
        };
        const std::regex re(R"(mtp\.(\d+)\.(.*))");
        for (const auto & [key, t] : ts) {
            std::smatch m;
            if (!std::regex_match(key, m, re)) continue;
            const std::string stage = m[1], rest = m[2];
            if (rest.size() > 6 && rest.compare(rest.size() - 6, 6, ".scale") == 0) continue;   // read with its weight
            if (rest == "ffn.gate.bias_vl") continue;   // the vision bias: text only here
            if (rest.rfind("ffn.experts.", 0) == 0) continue;   // stacked below
            const auto nm = kNames.find(rest);
            if (nm == kNames.end()) throw std::runtime_error("unexpected tensor " + key);
            const std::string name = "dspark." + stage + "." + nm->second;
            const St * scale = ts.count(key.substr(0, key.size() - 7) + ".scale") ? &ts.at(key.substr(0, key.size() - 7) + ".scale") : nullptr;
            if (t.shape.size() == 1 || (t.shape.size() == 2 && t.shape[0] == 1)) {   // vectors -> F32
                const int64_t n = t.shape.size() == 1 ? t.shape[0] : t.shape[1];
                std::vector<uint8_t> out(4 * n);
                St row = t;
                row.shape = {1, n};
                row_floats(row, nullptr, 0, reinterpret_cast<float *>(out.data()));
                add(name, GGML_TYPE_F32, {n}, std::move(out));
            } else if (t.dtype == "F8_E4M3") {   // FP8 matrices -> Q8_0
                if (!scale) throw std::runtime_error(key + ": FP8 without its scale");
                add(name, GGML_TYPE_Q8_0, {t.shape[1], t.shape[0]},
                    quantize(GGML_TYPE_Q8_0, t.shape[0], t.shape[1], [&](int64_t o, float * out) { row_floats(t, scale, o, out); }));
            } else if (t.dtype == "BF16") {   // the router, the Markov head: as they are
                add(name, GGML_TYPE_BF16, {t.shape[1], t.shape[0]}, std::vector<uint8_t>(t.data, t.data + t.bytes));
            } else if (t.dtype == "F32") {   // the hc mixes
                add(name, GGML_TYPE_F32, {t.shape[1], t.shape[0]}, std::vector<uint8_t>(t.data, t.data + t.bytes));
            } else {
                throw std::runtime_error(key + ": unexpected dtype " + t.dtype);
            }
        }
        // the routed experts: w1 (gate) / w3 (up) / w2 (down) of every expert, MXFP4 -> IQ4_XS, stacked
        const int n_exp = cfg.value("dspark_n_routed_experts", 0);
        double se = 0, sv = 0;
        for (int s = 0; s < n_stage; ++s)
            for (const auto & [w, gname] : std::vector<std::pair<std::string, std::string>>{
                     {"w1", "ffn_gate_exps"}, {"w3", "ffn_up_exps"}, {"w2", "ffn_down_exps"}}) {
                const std::string p = "mtp." + std::to_string(s) + ".ffn.experts.";
                const St & t0 = ts.at(p + "0." + w + ".weight");
                const int64_t rows = t0.shape[0], in = 2 * t0.shape[1];
                auto get = [&](int64_t r, float * out) {
                    const int64_t e = r / rows, o = r % rows;
                    const std::string k = p + std::to_string(e) + "." + w;
                    row_floats(ts.at(k + ".weight"), &ts.at(k + ".scale"), o, out);
                };
                std::vector<uint8_t> q = quantize(GGML_TYPE_IQ4_XS, rows * n_exp, in, get);
                {   // the error of one expert's rows (IQ4_XS against the MXFP4 values)
                    std::vector<float> x(in), y(in);
                    const size_t rb = ggml_row_size(GGML_TYPE_IQ4_XS, in);
                    const auto * tr = ggml_get_type_traits(GGML_TYPE_IQ4_XS);
                    for (int64_t o = 0; o < rows; o += 97) {
                        get(o, x.data());
                        tr->to_float(q.data() + rb * o, y.data(), in);
                        for (int64_t i = 0; i < in; ++i) { se += (x[i] - y[i]) * (x[i] - y[i]); sv += x[i] * x[i]; }
                    }
                }
                add("dspark." + std::to_string(s) + "." + gname + ".weight", GGML_TYPE_IQ4_XS, {in, rows, n_exp}, std::move(q));
            }
        std::printf("experts: relative RMS error of IQ4_XS against MXFP4 %.4f\n", std::sqrt(se / sv));
        if (!gguf_write_to_file(gg, argv[2], false)) throw std::runtime_error(std::string("cannot write ") + argv[2]);
        std::printf("wrote %s (%zu tensors)\n", argv[2], keep.size());
        gguf_free(gg);
        ggml_free(ctx);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "bl-dspark-pack: %s\n", e.what());
        return 1;
    }
    return 0;
}
