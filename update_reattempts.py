from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['skillbot_db']

# Update all existing test attempts to allow re-attempts
result = db.student_test_attempts.update_many(
    {},  # Match all documents
    {'$set': {'can_reattempt': True}}
)

print(f"Updated {result.modified_count} test attempts to allow re-attempts")
print(f"Matched {result.matched_count} total attempts")
