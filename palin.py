def is_palindrome(text):
    cleaned_text = ''.join(e for e in text if e.isalnum()).lower()
    return cleaned_text == cleaned_text[::-1]

print(is_palindrome("A man, a plan, a canal: Panama"))  # True
