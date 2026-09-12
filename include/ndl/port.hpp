// NDL v1.0 — compiler-side portability helpers (POSIX / Win32).
//
// One place for everything that used to assume Linux: mkdir(2), /proc/self/exe,
// isatty(3), getpid(2), shell quoting for sh vs cmd.exe and unique temp files.
// Only the TU that includes this header pays for <windows.h> (MinGW only);
// the helpers are inline, so no extra object file is needed in the build.

#pragma once

#include <algorithm>  // sort (listFiles)
#include <cstdio>
#include <cstdlib>  // getenv (self-contained regardless of TU include order)
#include <cstring>  // strlen (listFiles)
#include <string>
#include <vector>

#if defined(_WIN32)
  #if !defined(WIN32_LEAN_AND_MEAN)
    #define WIN32_LEAN_AND_MEAN
  #endif
  #if !defined(NOMINMAX)
    #define NOMINMAX
  #endif
  #include <windows.h>   // GetModuleFileNameA, FindFirstFileA
  #include <direct.h>    // _mkdir
  #include <io.h>        // _isatty
  #include <process.h>   // _getpid
#else
  #include <dirent.h>    // opendir/readdir (listFiles)
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <unistd.h>    // mkdir, readlink, isatty, getpid
#endif

namespace ndlport {

// --- single directory creation (no mkdir -p) ---------------------------------
inline bool makeDir(const std::string& path) {
#if defined(_WIN32)
  return ::_mkdir(path.c_str()) == 0;
#else
  return ::mkdir(path.c_str(), 0777) == 0;
#endif
}

// --- full path of the running executable --------------------------------------
// The result is normalized to forward slashes on Windows: every downstream
// path helper splits on '/', and all Win32/MSVCRT APIs accept '/' throughout.
inline std::string selfExePath() {
  char buf[4096];
#if defined(_WIN32)
  DWORD n = ::GetModuleFileNameA(nullptr, buf, sizeof(buf) - 1);
  if (n > 0 && n < sizeof(buf)) {
    buf[n] = '\0';
    std::string s(buf);
    for (char& c : s)
      if (c == '\\') c = '/';
    return s;
  }
  return std::string();
#else
  ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n > 0) {
    buf[n] = '\0';
    return std::string(buf);
  }
  return std::string();
#endif
}

// --- is stderr a terminal? -----------------------------------------------------
inline bool stderrIsTty() {
#if defined(_WIN32)
  return ::_isatty(2) != 0;  // CRT fd 2 == stderr on MinGW
#else
  return ::isatty(STDERR_FILENO) != 0;
#endif
}

// --- current process id --------------------------------------------------------
inline int pidNow() {
#if defined(_WIN32)
  return ::_getpid();
#else
  return static_cast<int>(::getpid());
#endif
}

// --- quote an argument for the platform shell (sh vs cmd.exe) ------------------
inline std::string shellQuote(const std::string& s) {
#if defined(_WIN32)
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') out += "\"\"";  // cmd.exe doubles embedded quotes
    else out += c;
  }
  out += '"';
  return out;
#else
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  out += '\'';
  return out;
#endif
}

// --- "discard stderr" redirection for the platform shell -----------------------
inline const char* devNullRedir() {
#if defined(_WIN32)
  return "2>NUL";
#else
  return "2>/dev/null";
#endif
}

// --- wrap a full command line for std::system()/_popen() -----------------------
// On Windows both run the line through `cmd.exe /c`. When the line STARTS with
// a quote, cmd's legacy parsing strips the first character and the LAST quote
// on the line -- so `"C:\Program Files\...\clang++.exe" args` becomes
// `C:\Program Files\... unquoted` and cmd then executes `C:\Program`.
// Wrapping the whole line in an extra quote pair neutralizes this: cmd strips
// exactly the added outer quotes and the original line passes through intact.
// POSIX runs the line via /bin/sh, which needs no such trick.
inline std::string wrapForCmd(const std::string& line) {
#if defined(_WIN32)
  return "\"" + line + "\"";
#else
  return line;
#endif
}

// --- directory listing (files only, filtered by extension) --------------------
// Returns bare file names (not paths) ending in `ext` (e.g. ".cpp"), sorted.
inline std::vector<std::string> listFiles(const std::string& dir, const char* ext) {
  std::vector<std::string> out;
#if defined(_WIN32)
  WIN32_FIND_DATAA fd;
  HANDLE h = ::FindFirstFileA((dir + "/*").c_str(), &fd);
  if (h == INVALID_HANDLE_VALUE) return out;
  do {
    if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
    std::string name = fd.cFileName;
    if (name.size() >= std::strlen(ext) &&
        name.compare(name.size() - std::strlen(ext), std::strlen(ext), ext) == 0)
      out.push_back(name);
  } while (::FindNextFileA(h, &fd));
  ::FindClose(h);
#else
  DIR* d = ::opendir(dir.c_str());
  if (d == nullptr) return out;
  struct dirent* de;
  const size_t extLen = std::strlen(ext);
  while ((de = ::readdir(d)) != nullptr) {
    if (de->d_type == DT_DIR) continue;
    std::string name = de->d_name;
    if (name.size() >= extLen &&
        name.compare(name.size() - extLen, extLen, ext) == 0)
      out.push_back(name);
  }
  ::closedir(d);
#endif
  std::sort(out.begin(), out.end());
  return out;
}

// --- unique temp file (mkstemp replacement, no /tmp assumption) ----------------
// Creates the file so the path is reserved; the caller owns deletion.
inline std::string makeTempFilePath(const char* prefix) {
  const char* envs[] = {::getenv("TMPDIR"), ::getenv("TEMP"), ::getenv("TMP")};
  std::string base;
  for (const char* e : envs) {
    if (e != nullptr && e[0] != '\0') { base = e; break; }
  }
  if (base.empty()) {
#if defined(_WIN32)
    base = ".";
#else
    base = "/tmp";
#endif
  }
  if (!base.empty() && base.back() == '/') base.pop_back();

  const int pid = pidNow();
  for (int i = 0; i < 4096; ++i) {
    std::string p = base + "/" + prefix + "-" + std::to_string(pid) + "-" +
                    std::to_string(i) + ".tmp";
    FILE* f = std::fopen(p.c_str(), "wb");
    if (f != nullptr) {
      std::fclose(f);
      return p;
    }
  }
  return std::string();
}

}  // namespace ndlport
