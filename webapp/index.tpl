<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="stylesheet" href="/static/style.css" />
    <title>adgression</title>
  </head>
  <body>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" accept="image/*" name="data">
        <input type="submit" value="Upload">
    </form>
    <div id="fileDrop"> 
      <div class="caption">Drop an image file here.</div>
      <div id="result"></div>
    </div>
    <script src="/static/app.js"></script>
  </body>
</html>

