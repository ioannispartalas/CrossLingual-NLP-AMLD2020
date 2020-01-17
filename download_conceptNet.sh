echo "Start downloading embeddings.."
wget "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz" 
echo "Embeddings downloaded. Start unzipping."
gunzip numberbatch-19.08.txt.gz
echo "Embeddings unzipped. Start filtering English and Spanish words."
python extract_embeddings.py
echo "Done. Cleaning.."
echo "Done!"
echo 
echo
echo 

