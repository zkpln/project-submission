#!/usr/bin/env pwsh

Invoke-WebRequest -Uri http://people.cs.pitt.edu/~nineil/crossmod/ads.zip
Expand-Archive -Path ./ads.zip -DestinationPath ./data
Remove-Item ads.zip
