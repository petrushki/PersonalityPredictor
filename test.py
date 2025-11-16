import requests

test_sample = {
    "time_spent_alone": 4, 
    "stage_fear": 1, 
    "social_event_attendance": 4,
    "going_outside": 6, 
    "drained_after_socializing": 1, 
    "friends_circle_size": 4,
    "post_frequency": 0
}

url = 'http://localhost:9696/predict'

response = requests.post(url, json=test_sample)
predictions = response.json()

print(predictions)
if predictions['introvert']:
    print("The personality is likely introvert")
else:
    print("The presonality is likely extrovert")