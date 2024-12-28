from django.contrib.auth.models import User

def username_present(username):
    """
    Check if a username exists in the database
    """
    if User.objects.filter(username=username).exists():
        return True
    return False