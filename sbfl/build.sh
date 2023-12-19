#!/bin/bash
set -e

pushd e9patch; CC=gcc CXX=g++ ./build.sh; popd;
cp hooks/src_tracer.c e9patch/examples;
cd e9patch; CC=gcc CXX=g++ ./e9compile.sh examples/src_tracer.c
