#~/bin/bash

echo "Welcome"
sleep 1
echo "Starting script"


# source ~/charles/environments/keras0/bin/activate
echo 'fp_qbdc_KL'
python  experiment.py 1 1 0
python  experiment.py 1 1 1

python  experiment.py 1 2 0
python  experiment.py 1 2 1

python  experiment.py 1 3 0
python  experiment.py 1 3 1

echo 'fqbdc jensen acquition'
python  experiment.py 2 1 0
python  experiment.py 2 1 1

python  experiment.py 2 2 0
python  experiment.py 2 2 1

python  experiment.py 2 3 0
python  experiment.py 2 3 1


echo 'qbdc'
python  experiment.py 3 1 0
python  experiment.py 3 1 1

python  experiment.py 3 2 0
python  experiment.py 3 2 1

python  experiment.py 3 3 0
python  experiment.py 3 3 1


echo 'bald'
python  experiment.py 4 1 0
python  experiment.py 4 1 1

python  experiment.py 4 2 0
python  experiment.py 4 2 1

python  experiment.py 4 3 0
python  experiment.py 4 3 1

echo 'Random'
python  experiment.py 5 1 0
python  experiment.py 5 1 1

python  experiment.py 5 2 0
python  experiment.py 5 2 1

python  experiment.py 5 3 0
python  experiment.py 5 3 1
