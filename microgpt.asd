;;;; microgpt.asd

(asdf:defsystem #:microgpt
  :description "Describe microgpt here"
  :author "Giovanni Crisalfi <giovanni.crisalfi@protonmail.com>"
  :license  "GPL-3.0-or-later"
  :version "0.0.1"
  :serial t
  :components ((:file "package")
               (:file "microgpt")
               (:static-file "input.txt")))
