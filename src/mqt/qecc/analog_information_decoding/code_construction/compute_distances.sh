#!/bin/bash
# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

path=$1

gap -q << EOF > $path/info.txt
LoadPackage("QDistRnd");;
hx:=ReadMTXE("$path/hx.mtx");;

hz:=ReadMTXE("$path/hz.mtx");;

Print("dZ = ", DistRandCSS(hx[3], hz[3], 150, 0,2), "\n");
Print("dX = ", DistRandCSS(hz[3], hx[3], 150, 0,2), "\n");

file := IO_File("$path/mx.txt");
lines := IO_ReadLines(file);
parityCheckMatrix := [];

for l in lines do
  row := ReplacedString(l, "\n", "");
  r := SplitString(row, " ");
  newr := [];
  for b in r do
    c := Int(b);
    Add(newr,c);
  od;
  Add(parityCheckMatrix, newr);
od;

xcode := CheckMatCode(parityCheckMatrix, GF(2));

file := IO_File("$path/mz.txt");
lines := IO_ReadLines(file);
parityCheckMatrix := [];

for l in lines do
  row := ReplacedString(l, "\n", "");
  r := SplitString(row, " ");
  newr := [];
  for b in r do
    c := Int(b);
    Add(newr,c);
  od;
  Add(parityCheckMatrix, newr);
od;

zcode := CheckMatCode(parityCheckMatrix, GF(2));



# Print("dMx = ", MinimumDistance(xcode), "\n"); only works for small codes quickly
Print("dMx = ", MinimumDistanceLeon(xcode), "\n"); #https://gap-packages.github.io/guava/doc/chap4_mj.html#X8170B52D7C154247:~:text=4.8%2D4%20MinimumDistanceLeon
Print("dMz = ", MinimumDistanceLeon(zcode), "\n");

EOF
