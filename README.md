# name-or-not
Check if given string is name or not

docker

```sh
sudo docker run -p 8000:8000 -h 0.0.0.0 pkumdev/nameornot

curl -X POST -H "Content-Type: application/json" -d '{"input_text": "Donald Trump"}' http://localhost:8000/predict

```
