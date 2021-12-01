binvox
=======

Binvox is a file-format developed by Patrick Min that helps to store
the 3D binary-volume (e.g. occupancy grid) in compact manner. 
This repository contains a library for loading &amp; saving the Binvox files.

A popular script to load/save binvox files has been published
at `here <https://github.com/dimatura/binvox-rw-py>`_. However, the code is
almost 10 years old now and every project duplicated the script file
when using it.

This project is based on the available code-base. The main aim is to
re-implement binvox file I/O as a library while taking OOP structure and
easy distribution (packaging) into account.

Installation
============

Pip can be used to install the library:

``pip install binvox``

Understanding the binvox file-format
====================================

Information regarding the format can be found here: `patrickmin.com/binvox <https://www.patrickmin.com/binvox/>`_
