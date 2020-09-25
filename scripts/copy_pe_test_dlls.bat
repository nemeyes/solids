@ECHO OFF
SETLOCAL

SET build_mode=""
SET work_dir=%1
ECHO %work_dir%

IF "%~2"=="--h" GOTO USAGE1
IF /i "%~2"=="Debug" set build_mode="Debug"
IF /i "%~2"=="Release" set build_mode="Release"

IF /i %build_mode%=="" set build_mode="Release"
IF /i %work_dir%=="" goto usage1

IF /i %build_mode%=="Release" (
	FC /b %work_dir%\3rdparty\gperf\v2.7\bin\libtcmalloc_minimal.dll %work_dir%\release\x64\%build_mode%\bin\PETest\libtcmalloc_minimal.dll > nul	
	IF errorLevel 1 (
		xcopy /Y %work_dir%\3rdparty\gperf\v2.7\bin\libtcmalloc_minimal.dll						%work_dir%\release\x64\%build_mode%\bin\PETest\
	)
)

FC /b %work_dir%\3rdparty\cudnn\v8.0.1\bin\cudnn_adv_infer64_8.dll %work_dir%\release\x64\%build_mode%\bin\PETest\cudnn_adv_infer64_8.dll > nul
IF  errorlevel 1 (
	xcopy /Y %work_dir%\3rdparty\cudnn\v8.0.1\bin\cudnn_adv_infer64_8.dll							%work_dir%\release\x64\%build_mode%\bin\PETest\
)

FC /b %work_dir%\3rdparty\cudnn\v8.0.1\bin\cudnn_cnn_infer64_8.dll %work_dir%\release\x64\%build_mode%\bin\PETest\cudnn_cnn_infer64_8.dll > nul
IF  errorlevel 1 (
	xcopy /Y %work_dir%\3rdparty\cudnn\v8.0.1\bin\cudnn_cnn_infer64_8.dll							%work_dir%\release\x64\%build_mode%\bin\PETest\
)

FC /b %work_dir%\3rdparty\cudnn\v8.0.1\bin\cudnn_ops_infer64_8.dll %work_dir%\release\x64\%build_mode%\bin\PETest\cudnn_ops_infer64_8.dll > nul
IF  errorlevel 1 (
	xcopy /Y %work_dir%\3rdparty\cudnn\v8.0.1\bin\cudnn_ops_infer64_8.dll							%work_dir%\release\x64\%build_mode%\bin\PETest\
)

FC /b %work_dir%\3rdparty\cudnn\v8.0.1\bin\cudnn64_8.dll %work_dir%\release\x64\%build_mode%\bin\PETest\cudnn64_8.dll > nul
IF  errorlevel 1 (
	xcopy /Y %work_dir%\3rdparty\cudnn\v8.0.1\bin\cudnn64_8.dll								%work_dir%\release\x64\%build_mode%\bin\PETest\
)

FC /b %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\myelin64_1.dll %work_dir%\release\x64\%build_mode%\bin\PETest\myelin64_1.dll > nul
IF  errorlevel 1 (
	xcopy /Y %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\myelin64_1.dll							%work_dir%\release\x64\%build_mode%\bin\PETest\
)

FC /b %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\nvinfer.dll %work_dir%\release\x64\%build_mode%\bin\PETest\nvinfer.dll > nul
IF  errorlevel 1 (
	xcopy /Y %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\nvinfer.dll								%work_dir%\release\x64\%build_mode%\bin\PETest\
)

FC /b %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\nvinfer_plugin.dll %work_dir%\release\x64\%build_mode%\bin\PETest\nvinfer_plugin.dll > nul
IF  errorlevel 1 (
	xcopy /Y %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\nvinfer_plugin.dll							%work_dir%\release\x64\%build_mode%\bin\PETest\
)

FC /b %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\nvonnxparser.dll %work_dir%\release\x64\%build_mode%\bin\PETest\nvonnxparser.dll > nul
IF  errorlevel 1 (
	xcopy /Y %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\nvonnxparser.dll							%work_dir%\release\x64\%build_mode%\bin\PETest\
)

FC /b %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\nvparsers.dll %work_dir%\release\x64\%build_mode%\bin\PETest\nvparsers.dll > nul
IF  errorlevel 1 (
	xcopy /Y %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\nvparsers.dll							%work_dir%\release\x64\%build_mode%\bin\PETest\
)

FC /b %work_dir%\release\x64\%build_mode%\bin\ThreadPoolManager.dll %work_dir%\release\x64\%build_mode%\bin\PETest\ThreadPoolManager.dll > nul
IF  errorlevel 1 (
	xcopy /Y %work_dir%\release\x64\%build_mode%\bin\ThreadPoolManager.dll							%work_dir%\release\x64\%build_mode%\bin\PETest\
	IF /i %build_mode%=="Debug" xcopy /Y %work_dir%\release\x64\%build_mode%\bin\ThreadPoolManager.pdb			%work_dir%\release\x64\%build_mode%\bin\PETest\
)

FC /b %work_dir%\release\x64\%build_mode%\bin\FFDemuxer.dll %work_dir%\release\x64\%build_mode%\bin\PETest\FFDemuxer.dll > nul
IF  errorlevel 1 (
	xcopy /Y %work_dir%\release\x64\%build_mode%\bin\FFDemuxer.dll								%work_dir%\release\x64\%build_mode%\bin\PETest\
	IF /i %build_mode%=="Debug" xcopy /Y %work_dir%\release\x64\%build_mode%\bin\FFDemuxer.pdb				%work_dir%\release\x64\%build_mode%\bin\PETest\
)


FC /b %work_dir%\release\x64\%build_mode%\bin\NVDecoder.dll %work_dir%\release\x64\%build_mode%\bin\PETest\NVDecoder.dll > nul
IF  errorlevel 1 (
	xcopy /Y %work_dir%\release\x64\%build_mode%\bin\NVDecoder.dll								%work_dir%\release\x64\%build_mode%\bin\PETest\
	IF /i %build_mode%=="Debug" xcopy /Y %work_dir%\release\x64\%build_mode%\bin\NVDecoder.pdb				%work_dir%\release\x64\%build_mode%\bin\PETest\
)

FC /b %work_dir%\release\x64\%build_mode%\bin\NVPoseEstimator.dll %work_dir%\release\x64\%build_mode%\bin\PETest\NVPoseEstimator.dll > nul
IF  errorlevel 1 (
	xcopy /Y %work_dir%\release\x64\%build_mode%\bin\NVPoseEstimator.dll							%work_dir%\release\x64\%build_mode%\bin\PETest\
	IF /i %build_mode%=="Debug" xcopy /Y %work_dir%\release\x64\%build_mode%\bin\NVPoseEstimator.pdb			%work_dir%\release\x64\%build_mode%\bin\PETest\
)

FC /b %work_dir%\release\x64\%build_mode%\bin\NVRenderer.dll %work_dir%\release\x64\%build_mode%\bin\PETest\NVRenderer.dll > nul
IF  errorlevel 1 (
	xcopy /Y %work_dir%\release\x64\%build_mode%\bin\NVRenderer.dll								%work_dir%\release\x64\%build_mode%\bin\PETest\
	IF /i %build_mode%=="Debug" xcopy /Y %work_dir%\release\x64\%build_mode%\bin\NVRenderer.pdb				%work_dir%\release\x64\%build_mode%\bin\PETest\
)

ECHO copy complited.
goto end

:: --------------------
:USAGE1
:: --------------------
ECHO usage: [DEBUG^|RELEASE] work_dir
ECHO.
ECHO copy necessary files to elastics static view folder.

:end