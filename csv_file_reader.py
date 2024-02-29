import csv
from datetime import datetime, timedelta

'''def time_diff(file_path):
    file_path= 'csv_' + file_path.replace('.mp4', '') + '.csv'
    with open(file_path, 'r') as file:
        # Move the file pointer to the end of the file
        file.seek(0, 2)
        
        # Find the position of the last newline character
        pos = file.tell()
        pos -= 3
        while pos > 0:
            pos -= 1
            file.seek(pos)
            if file.read(1) == '_':
                break
        
        # Read the last line
        last_line = file.readline().strip()
    #print(last_line)
    date_format = "%mM%dD%Hh%Mm"
    current_time = datetime.now()
    formatted_time = current_time.strftime(date_format)
    #print(formatted_time)


    # Convert strings to datetime objects
    date1 = datetime.strptime(formatted_time, date_format)
    date2 = datetime.strptime(last_line, date_format)
    # Calculate the time difference
    time_difference = date1 - date2
    print(time_difference)
    return(time_difference) 
'''

def time_diff(file_path):
    file_path= 'csv_' + file_path.replace('.mp4', '') + '.csv'
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        data = list(reader)  # Convert CSV rows into a list of lists

        if data:  # Check if there is any data in the CSV file
            last_row = data[-1]  # Get the last row
            if len(last_row) >= 3:  # Check if the last row has at least three values
                last_line = last_row[3]
    date_format = "%yY%mM%dD%Hh%Mm%Ss"
    current_time = datetime.now()
    last_time = last_line.replace("_", "").replace("2024", "24")
    formatted_time = current_time.strftime(date_format)

    # Convert strings to datetime objects
    date1 = datetime.strptime(formatted_time, date_format)
    date2 = datetime.strptime(last_time, date_format)
    # Calculate the time difference
    time_difference = date1 - date2
    return(time_difference)

def number_detections(file_path):
    file_path= 'csv_' + file_path.replace('.mp4', '') + '.csv'
    search_value = 10
    count = 0
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Assuming the first column contains integers
            if row and int(row[0]) == search_value:
                count += 1
    return (count)

def avg_detections(file_path):
    file_path= 'csv_' + file_path.replace('.mp4', '') + '.csv'
    avg=[]
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Assuming the first column contains integers
            avg.append(float(row[2]))
    return (avg)

def time_stopped(file_path):
    file_path= 'csv_' + file_path.replace('.mp4', '') + '.csv'
    summ = 0
    prev_row = None
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if (int(row[0]) == 1 and prev_row != None):
                # Assuming the first column contains integers
                if (int(prev_row[0]) > 10):
                    summ += float(prev_row[4])
            elif (int(row[0]) > 1):
                prev_row = row
    return (summ)

'''
file_path = 'csv_phone.csv'  # Replace 'your_file.csv' with the path to your CSV file
file_path = 'csv_172.17.4.131.csv'  # Replace 'your_file.csv' with the path to your CSV file
print(time_diff(file_path))
print(number_detections(file_path))
'''

