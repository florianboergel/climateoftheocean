# Climate of the ocean

build with jupyter books

Copy and paste your book’s _build contents into a new folder¶

The simplest way to host your book online is to simply copy everything that is inside _build and put it in a location where GitHub Pages knows to look. There are two places we recommend:

In a separate branch

    You can configure GitHub Pages to build any books that are in a branch that you specify. By default, this is gh-pages.
In a docs/ folder of your main branch

    If you’d like to keep your built book alongside your book’s source files, you may paste them into a docs/ folder.

    Warning

    Note that copying all of your book’s build files into the same branch as your source files will cause your repository to become very large over time, especially if you have many images in your book.

In either case, follow these steps:

    Copy the contents of _build/html directory into docs (or your other branch).

    Add a file called .nojekyll alongside your book’s contents. This tells GitHub Pages to treat your files as a “static HTML website”.

    Push your changes to GitHub, and configure it to start hosting your documentation.
