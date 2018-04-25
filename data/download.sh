#!/bin/bash          
# Prepare directory structure
mkdir scenes
mkdir 400x400
mkdir scenes/kinectv2
mkdir scenes/primesense
mkdir 400x400/primesense
mkdir 400x400/kinectv2

# Download scenes
for i in $(seq -f "%02g" 1 20)
do
  # Kinect
  cd scenes/kinectv2
  wget "http://ptak.felk.cvut.cz/darwin/t-less/v2/t-less_v2_test_kinect_$i.zip"
  unzip t-less_v2_test_kinect_$i.zip
  rm t-less_v2_test_kinect_$i.zip
  cd ..
  cd ..

  # Primesense
  cd scenes/primesense
  wget "http://ptak.felk.cvut.cz/darwin/t-less/v2/t-less_v2_test_primesense_$i.zip"
  unzip t-less_v2_test_primesense_$i.zip
  rm t-less_v2_test_primesense_$i.zip
  cd ..
  cd ..
done

# Download templates
for i in $(seq -f "%02g" 1 30)
do
  # Kinect
  cd 400x400/kinectv2
  wget "http://ptak.felk.cvut.cz/darwin/t-less/v2/t-less_v2_train_kinect_$i.zip"
  unzip t-less_v2_train_kinect_$i.zip
  rm t-less_v2_train_kinect_$i.zip
  cd ..
  cd ..

  # Primesense
  cd 400x400/primesense
  wget "http://ptak.felk.cvut.cz/darwin/t-less/v2/t-less_v2_train_primesense_$i.zip"
  unzip t-less_v2_train_primesense_$i.zip
  rm t-less_v2_train_primesense_$i.zip
  cd ..
  cd ..
done

# Run cleaning script
python clean_yml.py