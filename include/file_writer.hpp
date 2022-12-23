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
 */
class FileWriter {
 public:
  /**
   * @brief Construct a new File Writer object
   *
   * @param prefix File prefix
   * @param directory Directory to output
   */
  FileWriter(const std::string& prefix, const std::filesystem::path& directory)
      : prefix_{prefix}, directory_{directory} {
    std::filesystem::create_directory(directory);
  }

  /**
   * @brief Write data to a file.
   *
   * @tparam Derived
   * @param x Data
   * @param i Timestep
   *
   * The file name is "{prefix}{timestep}.txt".
   */
  template <typename Derived>
  void write(const Eigen::MatrixBase<Derived>& x, int i) const {
    namespace fs = std::filesystem;
    const std::string filename = fmt::format("{}{}.txt", prefix_, i);
    std::ofstream file(directory_ / fs::path(filename));
    file << x << std::endl;
  }

 private:
  std::string prefix_;               ///> Data file prefix
  std::filesystem::path directory_;  ///> Directory to output data
};

}  // namespace cfd

#endif  // CFD_FILE_WRITER_HPP