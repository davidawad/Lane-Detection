#!/bin/bash

rm test_images/*_output.png

for i in test_images/*; do python detection.py -f $i; done
