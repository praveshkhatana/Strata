@echo off
if exist "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
) else if exist "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
) else (
    echo ERROR: vcvars64.bat not found!
    exit /b 1
)
if errorlevel 1 exit /b 1

set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
if not exist "%CUDA_PATH%\bin\nvcc.exe" set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "GIT_CMD=C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\IDE\CommonExtensions\Microsoft\TeamFoundation\Team Explorer\Git\cmd"
set "PATH=%CUDA_PATH%\bin;C:\Program Files\CMake\bin;C:\Python312\Scripts;%GIT_CMD%;%PATH%"
if not exist "D:\Temp" mkdir "D:\Temp"
set "TEMP=D:\Temp"
set "TMP=D:\Temp"

if not exist "D:\Strata-v0.2.12\llamacpp\ggml\CMakeLists.txt" (
    echo Downloading pinned llama.cpp archive 3cf03257...
    mkdir "D:\Strata-v0.2.12\llamacpp" 2>nul
    curl.exe -sSL -o "D:\Temp\llamacpp.tar.gz" "https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.tar.gz"
    if errorlevel 1 exit /b 1
    tar.exe -xzf "D:\Temp\llamacpp.tar.gz" -C "D:\Strata-v0.2.12\llamacpp" --strip-components=1
    if errorlevel 1 exit /b 1
)

if exist "D:\Strata-v0.2.12\build\_deps" rd /s /q "D:\Strata-v0.2.12\build\_deps"

where cl nvcc cmake ninja git
if errorlevel 1 exit /b 1

cmake -S "D:\Strata-v0.2.12\repo" -B "D:\Strata-v0.2.12\build" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_GGML_DIR="D:/Strata-v0.2.12/llamacpp" ^
  -DSTRATA_WINDOW_MAX_T=16 ^
  -DSTRATA_BUILD_TESTS=ON ^
  -DSTRATA_ENABLE_CUDA=ON ^
  -DSTRATA_NATIVE_EXPERTS=ON ^
  -DSTRATA_PORTABLE=OFF ^
  -DSTRATA_MMQ_KQUANTS=ON ^
  -DSTRATA_Q6K_EXPERTS=OFF ^
  -DCMAKE_CUDA_ARCHITECTURES=89 ^
  -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -D__NV_NO_HOST_COMPILER_CHECK=1"
if errorlevel 1 exit /b 1

ninja -C "D:\Strata-v0.2.12\build" window_limits_test pool_test strata strata_kq -j 8
if errorlevel 1 exit /b 1

"D:\Strata-v0.2.12\build\window_limits_test.exe"
if errorlevel 1 exit /b 1
"D:\Strata-v0.2.12\build\pool_test.exe"
if errorlevel 1 exit /b 1

echo BUILD_AND_TESTS_PASSED
