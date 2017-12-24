@echo off

set CMAKE_GENERATOR="Visual Studio 14 2015 Win64"
set MSBUILD="%ProgramFiles(x86)%\MSBuild\14.0\Bin\amd64\MSBuild.exe"

mkdir build
cd build

cmake -G%CMAKE_GENERATOR% ..

%MSBUILD% /p:Configuration=Release gco.sln 