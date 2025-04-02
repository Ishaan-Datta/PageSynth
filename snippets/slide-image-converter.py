import pymupdf
import os

doc = pymupdf.open('example2.pdf')

for page in doc:
    pixmap = page.get_pixmap(dpi=300)
    pixmap.save(f"page-{page.number}.png")