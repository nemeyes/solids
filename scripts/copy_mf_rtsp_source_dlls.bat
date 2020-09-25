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
	FC /b %work_dir%\3rdparty\gperf\v2.7\bin\libtcmalloc_minimal.dll %work_dir%\release\x64\%build_mode%\bin\libtcmalloc_minimal.dll > nul	
	IF errorLevel 1 (
		xcopy /Y %work_dir%\3rdparty\gperf\v2.7\bin\libtcmalloc_minimal.dll				%work_dir%\release\x64\%build_mode%\bin\
	)
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