# Cochlear CI (Python)


## Install
```bash
python -m venv .venv && . .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# single file
python -m src.main data/input/voice.wav --out_dir data/output --N 8 --spacing log --kind fir --order 512 --lp_cut 400

#Run all folder
python -m src.main data/input --out_dir data/output --N 8 --spacing log --kind fir --order 512 --lp_cut 400