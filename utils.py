import xml

ALLOWED_PUNCTUATION = '.?!,'

class GroundTruthHandler(xml.sax.ContentHandler):

	def __init__(self, label):
		xml.sax.ContentHandler.__init__(self)
		self.label = label

	def startElement(self, name, attrs):
		if name == "article":
			self.label.append(attrs.getValue("hyperpartisan"))
