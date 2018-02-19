#pragma once


namespace dotnet_gui {

using namespace System;
using namespace System::IO;
using namespace System::Net;
using namespace System::Text;

public ref class HttpHelper
{
public:

	HttpHelper(String^ url)
	{
		const int bufSizeMax = 65536; // max read buffer size conserves memory
		const int bufSizeMin = 8192;  // min size prevents numerous small reads
		StringBuilder ^sb;

		// A WebException is thrown if HTTP request fails
		//try <-- no try, calling function must catch exception.
		{
			// Create an HttpWebRequest using WebRequest.Create (see .NET docs)!
			HttpWebRequest^ request = (dynamic_cast<HttpWebRequest^>(WebRequest::Create(url)));

			// Execute the request and obtain the response stream
			HttpWebResponse^ response = dynamic_cast<HttpWebResponse^>(request->GetResponse());
			Stream^ responseStream = response->GetResponseStream();

			// Content-Length header is not trustable, but makes a good hint.
			// Responses longer than int size will throw an exception here!
			int length = (int)response->ContentLength;

			// Use Content-Length if between bufSizeMax and bufSizeMin
			int bufSize = bufSizeMin;
			if (length > bufSize)
				bufSize = length > bufSizeMax ? bufSizeMax : length;


			// Allocate buffer and StringBuilder for reading response
			array<unsigned char> ^ buf = gcnew array<unsigned char> (bufSize);
			sb = gcnew StringBuilder(bufSize);

			// Read response stream until end
			while ((length = responseStream->Read(buf, 0, buf->Length)) != 0)
				sb->Append(Encoding::UTF8->GetString(buf, 0, length));

			responseString = sb->ToString();
		}
		/*catch (Exception ex)
		{
			sb = new StringBuilder(ex.Message);
		}*/
	}

	String ^ getValue(String ^name) {
		int start = responseString->IndexOf(name + L"{");
		int stop = -1;
		if(start >= 0) {
			start += name->Length + 1;
			stop = responseString->IndexOf(L'}', start);
		}

		if(start<0 || stop<0)
			return nullptr;
		else
			return responseString->Substring(start, stop-start);
	}

	String ^ getResponseStr() {
		return responseString;
	}

private:
	String^ responseString;
};

};
