'use strict';

var fileDropElement = document.getElementById("fileDrop");

function handleDragEnter(e) {
  e.preventDefault();
  e.target.classList.add("dragover");
  return true;
}
fileDropElement.addEventListener("dragenter", handleDragEnter, false);

function handleDragLeave(e) {
  e.target.classList.remove("dragover");
}
fileDropElement.addEventListener("dragleave", handleDragLeave, false);

function handleDragOver(e) {
  e.preventDefault();
  e.stopPropagation();
  return false;
}
fileDropElement.addEventListener("dragover", handleDragOver, false);

function handleDrop(e) {
  e.preventDefault();
  e.stopPropagation();
  handleDragLeave(e); // make sure the "drag and drop is over" event runs

  var resultElement = document.getElementById("result");

  var formData = new FormData();
  formData.append('data', e.dataTransfer.files[0]);
  fetch("/predict", {
    method: 'POST',
    body: formData,
  }).then(function (response) {
    return response.text();
  }).then(function (text ) {
    resultElement.innerText = text;
  }).catch(function (error) {
    alert("Error uploading image");
  });

  return false;
}
fileDropElement.addEventListener("drop", handleDrop, false);

