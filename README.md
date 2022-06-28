# Code reuse

Code implementation of https://www.cheminformania.com/master-your-molecule-generator-seq2seq-rnn-models-with-smiles-in-keras/

# Resources

1. Download the container: docker build -t <tag_name> .
2. Run the container: docker run -it --rm --gpus all --name pytorch -v $PWD:/work <tag_name> 
3. start jupyter notebook: jupyter notebook --ip 0.0.0.0 --no-browser --allow-root