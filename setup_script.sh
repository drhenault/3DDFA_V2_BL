# Set up virtual environment for FaceNet
echo "Initiating & setting up virtual environment for FaceNet"
cd facenet
python -m venv facenet-venv
source facenet-venv/bin/activate
pip install -r requirements-facenet-venv.txt
echo "Virtual environment for FaceNet initialized"
deactivate
cd ..

# Set up virtual environment for 3DDFA_v2
echo "Initiating & setting up virtual environment for 3DDFA_v2"
python -m venv 3ddfa-venv
source 3ddfa-venv/bin/activate
pip install -r requirements.txt
echo "Virtual environment for 3DDFA_v2 initialized"

# Run a build script from 3DDFA_v2
echo "Running Cython build script for 3DDFA_v2"
sh ./build.sh
deactivate

echo "All done!"
