cd data

wget https://fcp-indi.s3.us-east-1.amazonaws.com/data/Projects/INDI/ATLAS/R2.1/atlas21_training_raw.tar.gz 
wget https://fcp-indi.s3.us-east-1.amazonaws.com/data/Projects/INDI/ATLAS/R2.1/atlas21_training_preprocessed.tar.gz 

openssl aes-256-cbc -md sha256 -d -a -in atlas21_training_raw.tar.gz -out ATLAS_R2.1_raw.tar.gz
openssl aes-256-cbc -md sha256 -d -a -in atlas21_training_preprocessed.tar.gz -out ATLAS_R2.1_preprocessed.tar.gz