#!/bin/bash

# Set the URL of the zipped file
zip_url="https://storage.googleapis.com/tfhub-modules/google/universal-sentence-encoder/4.tar.gz"

# Set the desired filename for the downloaded file
zip_filename="sentence_encoder.zip"

# Download the zipped file
curl -o "$zip_filename" "$zip_url"

# Extract the contents of the zipped file in the root folder
unzip "$zip_filename" -d /

# Remove the downloaded zip file (optional)
rm "$zip_filename"
