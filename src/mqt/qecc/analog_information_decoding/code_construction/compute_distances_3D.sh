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

EOF
