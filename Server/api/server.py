from threading import Thread
from functools import partial
from urllib.parse import urlparse
from urllib.parse import parse_qs
from http.server import BaseHTTPRequestHandler, HTTPServer

import json
import logging
import base64


from api.post import Post
from api.put import Put
from api.get import Get

logging.basicConfig(level=logging.DEBUG)


"""
FloorplanToBlender3d
Copyright (C) 2021 Daniel Westberg
"""


class S(BaseHTTPRequestHandler):
    def __init__(self, shared, *args, **kwargs):
        self.shared = shared
        super().__init__(*args, **kwargs)

    def make_client(self):
        client = dict()
        client["address"] = self.address_string()
        client["port"] = str(self.client_address[1])
        return client

    def transform_dict(self, d):
        """Solve issue with all items are lists from query parser!"""
        res = dict()
        for key, item in d.items():
            res[key] = item[0]
        return res

    def query_parser(self, params, rmi):
        """Takes query dict, creates kwargs for methods"""
        function = params.get("func")
        if not function:
            logging.error("No function specified in parameters.")
            return None, None
        
        # Extract only the function name without query parameters
        function = function.split('?')[0]
        logging.debug(f"Function requested: {function}")
        
        out_rmi = rmi(client=self.make_client(), shared_variables=self.shared)
        try:
            func_obj = getattr(out_rmi, function)
            argc = func_obj.__code__.co_argcount
            args = func_obj.__code__.co_varnames[:argc]
        except AttributeError:
            logging.error(f"Function {function} not found in {rmi.__name__}")
            return None, None

        # Secure bad requests!
        if "__" in function or (argc == 0 and len(args) == 0):
            logging.error(f"Function {function} is invalid.")
            return None, None
        # # Generate set of correct variables!
        # kwargs = dict()

        # # If we want to use api_reference variables, these are ignored by swagger
        # if "_api_ref" in args:
        #     kwargs["_api_ref"] = self

        # if "_data" in args:
        #     kwargs["_data"] = params

        # if "func" in params:
        #     kwargs["func"] = params["func"]

        # # Add relevant data!
        # for parameter in params:
        #     if parameter in args:
        #         kwargs[parameter] = params[parameter]

        # Generate set of correct variables!
        kwargs = {key: value for key, value in params.items() if key in args}

        logging.debug(f"Function {function} will be called with arguments: {kwargs}")
        return out_rmi, kwargs


    def _set_response(self):
        self.send_response(200, "OK")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "POST, PUT, OPTIONS, HEAD, GET"
        )
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def do_HEAD(self):
        self._set_response()

    def do_OPTIONS(self):
        self._set_response()

    def do_GET(self):
        parsed_path = urlparse(self.path)
        kwargs = None
        try:
            parsed_data = self.transform_dict(parse_qs(parsed_path.query))
            rmi, kwargs = self.query_parser(parsed_data, Get)
        except Exception as e:
            message = "RECIEVED GET REQUEST WITH BAD QUERY: " + str(e)
            print(message)
        finally:
            if kwargs is None or rmi is None:
                message = "Function unavailable!"
            else:
                message = getattr(rmi, kwargs["func"])(**kwargs)
        try:
            self._set_response()
            self.wfile.write(bytes(message, encoding="utf-8"))
        except ConnectionAbortedError as e:
            return  # This occurs when server is sending file and client isn't waiting for extra message.

    def do_PUT(self):
        parsed_path = urlparse(self.path)
        parsed_data = self.transform_dict(parse_qs(parsed_path.query))
        kwargs = None
        ctype = self.headers.get("Content-Type")
        message = "Unknown error"

        logging.debug(f"Received PUT request with Content-Type: {ctype}")

        if ctype == "multipart/form-data":
            content_length = int(self.headers["Content-Length"])
            file = self.rfile.read(content_length)
            if file:
                try:
                    rmi, kwargs = self.query_parser(parsed_data, Put)
                    if kwargs is None or rmi is None:
                        message = "Function unavailable!"
                    else:
                        kwargs["file"] = file
                        (message, _) = getattr(rmi, kwargs["func"])(**kwargs)
                except ValueError as e:
                    message = f"RECEIVED PUT REQUEST WITH BAD DATA: {str(e)}"
                except KeyError as e:
                    message = f"KeyError: {str(e)}"
                except Exception as e:
                    message = f"Unknown error: {str(e)}"
            else:
                message = "NO FILE PROVIDED!"
        elif ctype in ["html/text", "json/application", "application/json", None]:
            try:
                content_length = int(self.headers["Content-Length"])
                if content_length > 0:
                    post_data = self.rfile.read(content_length)
                    logging.debug(f"Raw POST data: {post_data}")

                    data = json.loads(post_data)
                    logging.debug(f"Decoded JSON data: {data}")

                    file = data.get("file")
                    if file:
                        rmi, kwargs = self.query_parser(parsed_data, Put)
                        if kwargs is None or rmi is None:
                            message = "Function unavailable!"
                        else:
                            # Ensure file is a string before processing
                            if isinstance(file, bytes):
                                file = file.decode('utf-8')

                            # Decode base64 string if it contains the header
                            if file.startswith("data:image/png;base64,"):
                                file = file.replace("data:image/png;base64,", "")
                            elif file.startswith("data:image/jpeg;base64,"):
                                file = file.replace("data:image/jpeg;base64,", "")

                            file_bytes = base64.b64decode(file)
                            kwargs["file"] = file_bytes

                            message, _ = getattr(rmi, kwargs["func"])(**kwargs)
                    else:
                        message = "NO FILE PROVIDED IN JSON!"
                else:
                    message = "EMPTY PAYLOAD!"
            except json.JSONDecodeError as e:
                logging.error(f"JSONDecodeError: {e}")
                message = f"RECEIVED PUT REQUEST WITH BAD DATA: {str(e)}"
            except KeyError as e:
                logging.error(f"KeyError: {e}")
                message = f"KeyError: {str(e)}"
            except Exception as e:
                logging.error(f"Unknown error: {e}")
                message = f"Unknown error: {str(e)}"
        else:
            message = f"RECEIVED PUT REQUEST WITH BAD CTYPE: {str(ctype)}"

        logging.debug(f"Sending response: {message}")
        self._set_response()
        self.wfile.write(bytes(message, encoding="utf-8"))


class Server(Thread):
    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def run(self):
        server_address = (self.shared.restapiHost, int(self.shared.restapiPort))
        httpd = HTTPServer(server_address, partial(S, self.shared))
        try:
            print(
                "REST API SERVER up and serving at ",
                self.shared.restapiHost,
                self.shared.restapiPort,
            )
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        httpd.server_close()
