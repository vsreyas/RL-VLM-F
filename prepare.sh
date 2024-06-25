export PYTHONPATH=${PWD}:PYTHONPATH
export PYTHONPATH=${PWD}/softgym:$PYTHONPATH
export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
export MUJOCO_GL=egl
export GEMINI_API_KEY=$5AIzaSyAlqiZqc-uYSymSq40Pom4PCZndosBHkes
export OPENAI_API_KEY=$5sk-proj-949BygC19E83huSvkr4xT3BlbkFJwS60lvFDMhK4A2wTFeFM
conda activate rlvlmf1