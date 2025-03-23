from datetime import datetime

# Fetch the current date
current_date = datetime.now()

# Format the date as MM/DD/YYYY
formatted_date = current_date.strftime("%m/%d/%Y")

# Print the formatted date
print("Today's date is:", formatted_date)
