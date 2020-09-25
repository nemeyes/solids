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


IF /i %build_mode%=="Release" xcopy /Y %work_dir%\3rdparty\gperf\v2.7\bin\libtcmalloc_minimal.dll				%work_dir%\release\x64\%build_mode%\bin\PETest\

xcopy /Y %work_dir%\3rdparty\cudnn\v8.0.1\bin\cudnn_adv_infer64_8.dll								%work_dir%\release\x64\%build_mode%\bin\PETest\
xcopy /Y %work_dir%\3rdparty\cudnn\v8.0.1\bin\cudnn_cnn_infer64_8.dll								%work_dir%\release\x64\%build_mode%\bin\PETest\
xcopy /Y %work_dir%\3rdparty\cudnn\v8.0.1\bin\cudnn_ops_infer64_8.dll								%work_dir%\release\x64\%build_mode%\bin\PETest\
xcopy /Y %work_dir%\3rdparty\cudnn\v8.0.1\bin\cudnn64_8.dll									%work_dir%\release\x64\%build_mode%\bin\PETest\

xcopy /Y %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\myelin64_1.dll								%work_dir%\release\x64\%build_mode%\bin\PETest\
xcopy /Y %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\nvinfer.dll									%work_dir%\release\x64\%build_mode%\bin\PETest\
xcopy /Y %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\nvinfer_plugin.dll								%work_dir%\release\x64\%build_mode%\bin\PETest\
xcopy /Y %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\nvonnxparser.dll								%work_dir%\release\x64\%build_mode%\bin\PETest\
xcopy /Y %work_dir%\3rdparty\TensorRT\v7.1.3.4\lib\nvparsers.dll								%work_dir%\release\x64\%build_mode%\bin\PETest\

xcopy /Y %work_dir%\release\x64\%build_mode%\bin\ThreadPoolManager.dll								%work_dir%\release\x64\%build_mode%\bin\PETest\
IF /i %build_mode%=="Debug" xcopy /Y %work_dir%\release\x64\%build_mode%\bin\ThreadPoolManager.pdb				%work_dir%\release\x64\%build_mode%\bin\PETest\

xcopy /Y %work_dir%\release\x64\%build_mode%\bin\FFDemuxer.dll									%work_dir%\release\x64\%build_mode%\bin\PETest\
IF /i %build_mode%=="Debug" xcopy /Y %work_dir%\release\x64\%build_mode%\bin\FFDemuxer.pdb					%work_dir%\release\x64\%build_mode%\bin\PETest\

xcopy /Y %work_dir%\release\x64\%build_mode%\bin\NVDecoder.dll									%work_dir%\release\x64\%build_mode%\bin\PETest\
IF /i %build_mode%=="Debug" xcopy /Y %work_dir%\release\x64\%build_mode%\bin\NVDecoder.pdb					%work_dir%\release\x64\%build_mode%\bin\PETest\

xcopy /Y %work_dir%\release\x64\%build_mode%\bin\NVPoseEstimator.dll								%work_dir%\release\x64\%build_mode%\bin\PETest\
IF /i %build_mode%=="Debug" xcopy /Y %work_dir%\release\x64\%build_mode%\bin\NVPoseEstimator.pdb				%work_dir%\release\x64\%build_mode%\bin\PETest\

xcopy /Y %work_dir%\release\x64\%build_mode%\bin\NVRenderer.dll									%work_dir%\release\x64\%build_mode%\bin\PETest\
IF /i %build_mode%=="Debug" xcopy /Y %work_dir%\release\x64\%build_mode%\bin\NVRenderer.pdb					%work_dir%\release\x64\%build_mode%\bin\PETest\

ECHO copy complited.
goto end

:: --------------------
:USAGE1
:: --------------------
ECHO usage: [DEBUG^|RELEASE] work_dir
ECHO.
ECHO copy necessary files to elastics static view folder.

:end