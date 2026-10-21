#!/usr/bin/env bash
# Instructions to obtain the four benchmark datasets used in A3.
# This script does NOT auto-download (most datasets require manual registration).

echo "=== CDnet2014 ==="
echo "Register and download from: http://changedetection.net"
echo "After download, place the 'dataset/' folder at: \$CDNET_ROOT"
echo ""

echo "=== BMC ==="
echo "Download from: http://bmc.iut-auvergne.com"
echo "After download, place at: \$BMC_ROOT"
echo ""

echo "=== LASIESTA ==="
echo "Download from: https://www.gti.ssr.upm.es/data/LASIESTA"
echo "After download, place at: \$LASIESTA_ROOT"
echo ""

echo "=== VIRAT ==="
echo "Download from: https://viratdata.org"
echo "After download, place at: \$VIRAT_ROOT"
echo ""

echo "Set these paths in your config files under splits/ or pass --dataset-root."
