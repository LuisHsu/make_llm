// engine.hpp 的比對測試：每個 case 用 mg::Value 算一次，和 tools/gen_golden.py
// 以 PyTorch float64 產生的 golden 值比對輸出與各輸入的梯度。
//
// 介面約定（include/engine.hpp，namespace mg）：
//   - Value(double)：葉節點
//   - 運算子：+ - * /、一元負號，兩側都可以是 double（例如 2.0 * v、1.0 - v）
//   - member function：v.tanh()、v.relu()、v.exp()、v.pow(double)
//   - v.data()、v.grad() 回傳 double
//   - 對輸出呼叫 v.backward()
//   - 複製 Value 共用同一個節點（shared_ptr 語意）
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "engine.hpp"

namespace {

using Builder = std::function<mg::Value(std::vector<mg::Value>&)>;

// 每個 case 的運算式，和 gen_golden.py 的 CASES 一一對應。
const std::map<std::string, Builder>& Builders() {
  static const std::map<std::string, Builder> builders = {
      {"add", [](auto& v) { return v[0] + v[1]; }},
      {"mul", [](auto& v) { return v[0] * v[1]; }},
      {"sub", [](auto& v) { return v[0] - v[1]; }},
      {"div", [](auto& v) { return v[0] / v[1]; }},
      {"neg", [](auto& v) { return -v[0]; }},
      {"pow_int", [](auto& v) { return v[0].pow(3.0); }},
      {"pow_half", [](auto& v) { return v[0].pow(0.5); }},
      {"pow_negative", [](auto& v) { return v[0].pow(-2.5); }},
      {"scalar_right", [](auto& v) { return (v[0] + 1.0) * 2.0 - 3.0 / 4.0; }},
      {"scalar_left", [](auto& v) { return 1.0 + 2.0 * v[0]; }},
      {"scalar_rsub", [](auto& v) { return 1.0 - v[0]; }},
      {"scalar_rtruediv", [](auto& v) { return 3.0 / v[0]; }},
      {"tanh_negative", [](auto& v) { return v[0].tanh(); }},
      {"tanh_small", [](auto& v) { return v[0].tanh(); }},
      {"tanh_large", [](auto& v) { return v[0].tanh(); }},
      {"relu_negative", [](auto& v) { return v[0].relu(); }},
      {"relu_positive", [](auto& v) { return v[0].relu(); }},
      // x = 0 不可微，約定取 0（與 PyTorch 一致）
      {"relu_zero", [](auto& v) { return v[0].relu(); }},
      {"exp_negative", [](auto& v) { return v[0].exp(); }},
      {"exp_zero", [](auto& v) { return v[0].exp(); }},
      {"exp_positive", [](auto& v) { return v[0].exp(); }},
      {"reuse_add", [](auto& v) { return v[0] + v[0]; }},
      {"reuse_mul", [](auto& v) { return v[0] * v[0]; }},
      {"shared_intermediate",
       [](auto& v) {
         mg::Value c = v[0] * v[1];
         mg::Value d = c + v[0];
         mg::Value e = c * d;
         return e + d.tanh() - c;
       }},
      {"deep_composite",
       [](auto& v) {
         mg::Value x = v[0] * v[1] + v[2];
         mg::Value y = x.tanh() * v[0] - v[1].pow(2.0);
         mg::Value z = (y + x).relu() + (x * y).exp() / (v[2].pow(2.0) + 1.0);
         mg::Value w = z * z + 2.0 * y - x / 3.0;
         return w.tanh() + w * 0.1;
       }},
  };
  return builders;
}

nlohmann::json LoadGolden() {
  const std::filesystem::path path =
      std::filesystem::path(GOLDEN_DIR) / "engine_cases.json";
  EXPECT_TRUE(std::filesystem::exists(path))
      << "找不到 " << path
      << "，請先執行 uv run python 01_micrograd/cpp/tools/gen_golden.py";
  std::ifstream in(path);
  nlohmann::json j;
  in >> j;
  return j;
}

class EngineCase : public testing::TestWithParam<nlohmann::json> {};

TEST_P(EngineCase, MatchesPyTorch) {
  const nlohmann::json& c = GetParam();
  const auto name = c.at("name").get<std::string>();
  const auto inputs = c.at("inputs").get<std::vector<double>>();
  const auto want_grads = c.at("grads").get<std::vector<double>>();
  const double want_out = c.at("out").get<double>();

  const auto it = Builders().find(name);
  ASSERT_NE(it, Builders().end()) << "沒有對應的運算式：" << name;

  std::vector<mg::Value> vs;
  vs.reserve(inputs.size());
  for (const double x : inputs) {
    vs.emplace_back(x);
  }

  mg::Value out = it->second(vs);
  EXPECT_NEAR(out.data(), want_out, 1e-9 * std::max(1.0, std::abs(want_out)))
      << name << "：" << c.at("expr").get<std::string>();

  out.backward();
  ASSERT_EQ(vs.size(), want_grads.size()) << name;
  for (size_t i = 0; i < vs.size(); ++i) {
    EXPECT_NEAR(vs[i].grad(), want_grads[i],
                1e-9 * std::max(1.0, std::abs(want_grads[i])))
        << name << " 的第 " << i << " 個輸入：" << c.at("expr").get<std::string>();
  }
}

INSTANTIATE_TEST_SUITE_P(
    Golden, EngineCase, testing::ValuesIn([] {
      // 注意：LoadGolden() 回傳暫時物件，必須先綁到具名變數再走訪，
      // 否則 range-for 會抓到已經解構的 json
      const nlohmann::json golden = LoadGolden();
      std::vector<nlohmann::json> cases;
      for (const auto& c : golden.at("cases")) {
        cases.push_back(c);
      }
      return cases;
    }()),
    [](const testing::TestParamInfo<nlohmann::json>& info) {
      return info.param.at("name").get<std::string>();
    });

// golden 檔與運算式表必須一一對應，避免有 case 安靜地沒被測到
TEST(EngineCases, GoldenAndBuildersMatch) {
  const nlohmann::json golden = LoadGolden();
  std::set<std::string> golden_names;
  for (const auto& c : golden.at("cases")) {
    golden_names.insert(c.at("name").get<std::string>());
  }
  std::set<std::string> builder_names;
  for (const auto& [name, _] : Builders()) {
    builder_names.insert(name);
  }
  EXPECT_EQ(golden_names, builder_names);
}

// 每個 Value 都要能各自讀到自己的梯度（複製共用節點，不是各自獨立）
TEST(EngineSemantics, CopiesShareTheSameNode) {
  mg::Value a{2.0};
  mg::Value b = a;
  mg::Value out = a * 3.0;
  out.backward();
  EXPECT_EQ(a.data(), b.data());
  EXPECT_EQ(a.grad(), b.grad());
  EXPECT_DOUBLE_EQ(a.grad(), 3.0);
}

}  // namespace
