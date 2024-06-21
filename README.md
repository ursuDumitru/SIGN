# Pașii de rulare a programului

1. Se descarcă proiectul de pe GitHub.
```bash
git clone https://github.com/ursuDumitru/SIGN.git
```

2. Se navighează în directorul sistemului.
```bash
cd "calea_catre_directorul_proiectului/SIGN"
```

3. Se creează două medii virtuale python cu conda și se activează.
```bash
conda create -n sign_camera python=3.10
conda create -n sign_model python=3.10
```

4. Se activează mediul pentru cameră și se instalează dependințele.
```bash
conda activate sign_camera
cd ./src/camera_code/
pip install -r requirements.txt
```

5. Se activează mediul pentru model și se instalează dependințele.
```bash
conda activate sign_model
cd ./src/model_code/
pip install -r requirements.txt
```

6. Pentru a rula codul de identificare și stocare a semnelor se rulează scriptul app.py
```bash
cd ./src/model_code/
python app.py
```

7. Pentru a rula codul de antrenare a modelelor se rulează scripturile următoare:
```bash
cd ./src/model_code/
python train_dynamic_model.py
python train_static_model.py
```