video_not_found = [
    'Mjg0MDgxODgzMDUyOTk3NjE2OA==', # P1
    'LTI3NTA5MjY3MDQwMTgwMDg5MjY=', # P2
    'NjM0NTkzODMwMTIzMDAyOTcwNg==',
    'NzIyODk0Mjc2MTE4OTM3NTk4',
    'ODU0NjMxNTczMDI0MTQwMjE5MA=='] 
#LTgzMTE5MTE0Mjk0NDIwMzAxNzc=

def video_sanity_check(video_id):
    return video_id not in video_not_found