// golden 檔本身的健檢：確認 tools/gen_golden.py 與 tools/gen_nn_golden.py 產生的檔案
// 存在、能解析、欄位齊全。這裡不碰 engine.hpp / nn.hpp，所以實作還沒寫完時也跑得動。
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

nlohmann::json LoadGolden() {
  const std::filesystem::path path =
      std::filesystem::path(GOLDEN_DIR) / "engine_cases.json";
  // 檔案不存在時直接失敗，而不是安靜跳過
  EXPECT_TRUE(std::filesystem::exists(path))
      << "找不到 " << path << "，請先執行 uv run python 01_micrograd/cpp/tools/gen_golden.py";
  std::ifstream in(path);
  EXPECT_TRUE(in.is_open()) << "無法開啟 " << path;
  nlohmann::json j;
  in >> j;
  return j;
}

TEST(GoldenFile, ParsesAndHasCases) {
  const nlohmann::json j = LoadGolden();
  EXPECT_EQ(j.at("dtype").get<std::string>(), "float64");
  ASSERT_TRUE(j.at("cases").is_array());
  EXPECT_GT(j.at("cases").size(), 0u);
}

TEST(GoldenFile, EveryCaseIsWellFormed) {
  const nlohmann::json j = LoadGolden();
  std::set<std::string> names;
  for (const auto& c : j.at("cases")) {
    const auto name = c.at("name").get<std::string>();
    EXPECT_TRUE(names.insert(name).second) << "case 名稱重複：" << name;
    EXPECT_FALSE(c.at("expr").get<std::string>().empty()) << name;

    const auto inputs = c.at("inputs").get<std::vector<double>>();
    const auto grads = c.at("grads").get<std::vector<double>>();
    EXPECT_FALSE(inputs.empty()) << name;
    EXPECT_EQ(inputs.size(), grads.size()) << name;
    EXPECT_TRUE(std::isfinite(c.at("out").get<double>())) << name;
    for (const double g : grads) {
      EXPECT_TRUE(std::isfinite(g)) << name;
    }
  }
}

nlohmann::json LoadNnGolden() {
  const std::filesystem::path path = std::filesystem::path(GOLDEN_DIR) / "nn_cases.json";
  EXPECT_TRUE(std::filesystem::exists(path))
      << "找不到 " << path
      << "，請先執行 uv run python 01_micrograd/cpp/tools/gen_nn_golden.py";
  std::ifstream in(path);
  EXPECT_TRUE(in.is_open()) << "無法開啟 " << path;
  nlohmann::json j;
  in >> j;
  return j;
}

TEST(NnGoldenFile, ParsesAndHasCases) {
  const nlohmann::json j = LoadNnGolden();
  EXPECT_EQ(j.at("dtype").get<std::string>(), "float64");
  // 非線性函數換掉的話 golden 值就不再是正確答案，測試要跟著改
  EXPECT_EQ(j.at("nonlinearity").get<std::string>(), "tanh");
  ASSERT_TRUE(j.at("cases").is_array());
  EXPECT_GT(j.at("cases").size(), 0u);
}

TEST(NnGoldenFile, EveryCaseIsWellFormed) {
  const nlohmann::json j = LoadNnGolden();
  std::set<std::string> names;
  for (const auto& c : j.at("cases")) {
    const auto name = c.at("name").get<std::string>();
    EXPECT_TRUE(names.insert(name).second) << "case 名稱重複：" << name;

    const auto kind = c.at("kind").get<std::string>();
    EXPECT_TRUE(kind == "neuron" || kind == "layer" || kind == "mlp") << name << "：" << kind;

    const auto nin = c.at("nin").get<size_t>();
    const auto nouts = c.at("nouts").get<std::vector<size_t>>();
    ASSERT_FALSE(nouts.empty()) << name;
    if (kind != "mlp") {
      EXPECT_EQ(nouts.size(), 1u) << name << "：neuron / layer 只有一層";
    }

    // 參數個數：每層每個 neuron 有「上一層輸出數」個權重加 1 個 bias
    size_t want_params = 0;
    size_t prev = nin;
    for (const size_t n : nouts) {
      want_params += n * (prev + 1);
      prev = n;
    }
    const auto params = c.at("params").get<std::vector<double>>();
    const auto grads = c.at("param_grads").get<std::vector<double>>();
    EXPECT_EQ(params.size(), want_params) << name;
    EXPECT_EQ(grads.size(), want_params) << name;

    const auto inputs = c.at("inputs").get<std::vector<std::vector<double>>>();
    const auto targets = c.at("targets").get<std::vector<std::vector<double>>>();
    const auto outputs = c.at("outputs").get<std::vector<std::vector<double>>>();
    ASSERT_FALSE(inputs.empty()) << name;
    ASSERT_EQ(targets.size(), inputs.size()) << name;
    ASSERT_EQ(outputs.size(), inputs.size()) << name;
    for (size_t s = 0; s < inputs.size(); ++s) {
      EXPECT_EQ(inputs[s].size(), nin) << name << " 第 " << s << " 筆";
      EXPECT_EQ(targets[s].size(), nouts.back()) << name << " 第 " << s << " 筆";
      EXPECT_EQ(outputs[s].size(), nouts.back()) << name << " 第 " << s << " 筆";
      for (const double o : outputs[s]) {
        EXPECT_TRUE(std::isfinite(o)) << name;
      }
    }

    EXPECT_TRUE(std::isfinite(c.at("loss").get<double>())) << name;
    // 梯度全是 0 代表這個 case 測不到任何東西
    bool any_nonzero = false;
    for (const double g : grads) {
      EXPECT_TRUE(std::isfinite(g)) << name;
      any_nonzero = any_nonzero || g != 0.0;
    }
    EXPECT_TRUE(any_nonzero) << name << "：所有參數梯度都是 0";
  }
}

}  // namespace
