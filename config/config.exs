import Config

# minijinja_ex comes from git until a release has the Hugging Face compatibility. Its
# precompiled NIFs are older, so it builds from source. It reads only this variable.
System.put_env("MINIJINJA_EX_BUILD", "1")
