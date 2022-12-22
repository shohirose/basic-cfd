#ifndef CFD_FILE_WRITER_HPP
#define CFD_FILE_WRITER_HPP

#include <fmt/format.h>

#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <string>

namespace cfd {

/**
 * @brief Write data to a file in a specified directory
 *
 * @tparam TimestepChecker Checks timesteps writing data to a file
 */
template <typename TimestepChecker>
class FileWriter {
 public:
  /**
   * @brief Construct a new File Writer object
   *
   * @param prefix File prefix
   * @param directory Directory to output
   */
  FileWriter(const std::string& prefix, const std::filesystem::path& directory)
      : prefix_{prefix}, directory_{directory}, checker_{} {
    std::filesystem::create_directory(directory);
  }

  /**
   * @brief Construct a new File Writer object
   *
   * @param prefix File prefix
   * @param directory Directory to output
   * @param checker Timestep checker
   */
  FileWriter(const std::string& prefix, const std::filesystem::path& directory,
             const TimestepChecker& checker)
      : prefix_{prefix}, directory_{directory}, checker_{checker} {
    std::filesystem::create_directory(directory);
  }

  /**
   * @brief Print data to a file "{prefix}_{timestep}.txt"
   *
   * @tparam Derived
   * @param x Data
   * @param i Timestep
   */
  template <typename Derived>
  void operator()(const Eigen::MatrixBase<Derived>& x, int i) const {
    namespace fs = std::filesystem;
    if (checker_(i)) {
      const std::string filename = fmt::format("{}_{}.txt", prefix_, i);
      std::ofstream file(directory_ / fs::path(filename));
      file << x << std::endl;
    }
  }

 private:
  std::string prefix_;               ///> Data file prefix
  std::filesystem::path directory_;  ///> Directory to output data
  TimestepChecker checker_;          ///> Checks timestep to print
};

}  // namespace cfd

#endif  // CFD_FILE_WRITER_HPP