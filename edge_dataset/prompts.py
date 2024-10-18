bbox_formats = ["point", "bbox"]
def add_bbox_suffix(prompt, bbox_format):
    if bbox_format == "point":
        suffix = " (with point (x, y))"
    elif bbox_format == "bbox":
        suffix = " (with bbox (x1, x2, y1, y2))"
    else:
        raise ValueError(f"Invalid bbox_format: {bbox_format}!")
    return prompt[:-1] + suffix + prompt[-1]


basic = {
    "grounding": [
        "In the screenshot of this web page, please give me the coordinates of the element I want to click on according to my instructions.",
        "Based on the screenshot of the page, I give a text description and you give its corresponding location.",
        "In the image above, I will give a series of descriptions of the elements to be clicked. Please predict where you want to click.",
        "I will give textual descriptions of certain elements in the screenshot. Please predict the location of the corresponding element.",
        "Please identify the coordinates of the webpage elements I describe based on the provided screenshot.",
        "Given a screenshot, I will describe specific elements; your task is to predict their locations.",
        "Using the image of this webpage, can you determine the coordinates of the elements I describe?",
        "In this webpage capture, I will describe certain elements. Please locate them for me.",
        "I'll provide textual descriptions of elements in this webpage screenshot. Can you find their coordinates?",
        "From the given webpage screenshot, I need you to identify the locations of described elements.",
        "Based on this screenshot, I'll describe some elements. Please pinpoint their exact locations.",
        "For the elements I describe in this page capture, can you predict their positions?",
        "I will describe elements from a webpage screenshot; your role is to locate them.",
        "Using the attached screenshot of a webpage, please find the coordinates of described elements.",
        "From the image of this webpage, I will describe elements for you to locate.",
        "I'll give descriptions of certain webpage elements; please identify where they are in this screenshot.",
        "On this webpage screenshot, I will point out some elements; please predict their exact coordinates.",
        "In this web page image, please locate the elements as I describe them.",
        "Given this screenshot of a webpage, I'll describe some elements; locate them for me.",
        "Please use the provided webpage screenshot to locate the elements I describe.",
        "In the provided web page image, I'll describe specific elements. Identify their locations, please.",
        "With this screenshot of a webpage, can you locate the elements I describe?",
        "I will describe features on this webpage screenshot; please predict their positions.",
        "Using the screenshot of this webpage, identify the coordinates of elements I describe.",
        "On this webpage capture, I'll point out specific elements for you to locate.",
        "Please determine the location of elements I describe in this webpage screenshot.",
        "I'll describe certain elements on this webpage image; your task is to find their locations.",
        "Using this webpage screenshot, I'll describe some elements. Please locate them.",
        "Based on my descriptions, find the locations of elements in this webpage screenshot.",
        "In this web page capture, please predict the positions of elements I describe.",
        "I'll give textual clues about elements in this webpage screenshot; identify their coordinates.",
        "Using the provided screenshot, I'll describe webpage elements for you to locate.",
        "From this webpage image, I will describe specific elements. Please predict their exact locations."
    ],
    "ocr": [
        "Based on the screenshot of the web page, I give you the location to click on and you predict the text content of the corresponding element.",
        "In the image above, I give a series of coordinates and ask you to describe the corresponding elements.",
        "On this page, I will give you a series of coordinates and ask you to predict the text of the clickable element that corresponds to these coordinates.",
        "Given a webpage screenshot, I provide coordinates; predict the text content of the elements at these locations.",
        "In this screenshot, I'll give coordinates and ask you to describe the text of the elements there.",
        "Using the provided image of the webpage, I'll specify locations; you predict the text content of those elements.",
        "With this webpage capture, I provide a series of coordinates; please identify the text content of each element.",
        "In this page image, I'll point to specific locations; you need to predict the text of the corresponding elements.",
        "From this screenshot, I'll give coordinates; can you describe the text of the elements at these points?",
        "Based on this web page screenshot, I provide coordinates; please predict the textual content at these spots.",
        "Using the given image of the webpage, I'll specify certain coordinates; describe the text of the elements there.",
        "On this captured webpage, I will give a series of coordinates; your task is to predict the text at these locations.",
        "With this webpage image, I provide coordinates; can you tell me the text of the elements at these points?",
        "In the provided webpage screenshot, I'll point out locations; please describe the text of the elements there.",
        "From this web page capture, I give specific coordinates; predict the text content of the elements at these locations.",
        "Using this screenshot of a webpage, I'll indicate coordinates; can you predict the text of the elements?",
        "On this image of a web page, I provide coordinates; you need to describe the text of the corresponding elements.",
        "Given this webpage capture, I'll specify locations; please predict the text content of the elements there.",
        "In this screenshot, I give a series of coordinates; your task is to predict the text content of the elements.",
        "From the given webpage image, I'll provide coordinates; can you describe the text of the elements at these points?",
        "On this captured webpage, I provide specific coordinates; you need to predict the text of the elements there.",
        "Using this web page screenshot, I'll indicate locations; please describe the text content of the elements.",
        "With this image of a webpage, I specify coordinates; your task is to predict the text of the corresponding elements.",
        "In this webpage capture, I'll give coordinates; can you predict the text content of the elements at these locations?",
        "Based on this screenshot, I provide a series of coordinates; describe the text of the elements there.",
        "Using the image of this webpage, I'll specify locations; you need to predict the text of the elements.",
        "On this page screenshot, I give coordinates; please predict the text content of the corresponding elements.",
        "From this webpage image, I'll indicate specific coordinates; can you describe the text of the elements?",
        "In this web page image, I provide coordinates; your task is to predict the text of the elements at these locations.",
        "Given this screenshot of a webpage, I specify locations; please describe the text of the elements there.",
        "Using the provided page image, I'll point to locations; you predict the text content of the elements.",
        "On this webpage capture, I provide a series of coordinates; can you predict the text of the elements?",
        "With this image of the web page, I give specific coordinates; your task is to describe the text of the elements at these points."
    ]
}

accessibility = {
    "general_acb": basic["grounding"],
    "image_alt": [
        "Given this screenshot of a webpage, I specify locations of some images; please describe these images there.",
        "From the given webpage image, I'll provide coordinates of some images; can you describe their main content or role in the webpage at these points?",
        "On this captured webpage, I provide specific image coordinates; you need to predict the content of the images there.",
        "Using this web page screenshot, I'll indicate some images' locations; please summarize these images.",
        "With this image of a webpage, I specify coordinates of images; your task is to generate a brief description.",
        "In this webpage capture, I'll give coordinates of some images; can you briefly introduce their content or function?",
        "Based on this screenshot, I provide a series of coordinates of some images; describe the content there.",
    ]
}

captioning = {
    "title": [
        "Generate an appropriate title for this webpage, as a description of its content or a promotion slogan of its product.",
        "What could be a suitable title for this webpage to be displayed in search engine results, considering its content and functions?",
        "Observe this screen carefully, and then summarize it content or functionality of this webpage as a title.",
        "Generate an appropriate title for this webpage.",
        "What could be a suitable title for this webpage?",
        "How to summarize this webpage as a title?"
    ],
    "description": [
        "When using search engines to find this webpage, what may be the summary sentence displayed below the webpage title?",
        "Can you summarize the general content or functionality of this webpage in one sentence in order to display its content to users in search engine results?",
        "Generate a brief introduction or description of this webpage for future appearance in the summary of the search result below the webpage title.",
        "Generate a introduction or description of this webpage.",
        "Can you summarize this webpage as one sentence?"
    ],
    "keywords": [
        "Can you provide some keywords to describe the general content or functionality of this webpage?",
        "What keywords should I enter to find this webpage on a search engine?",
        "Please tell me some keywords to summarize the functionality or content of this webpage, which will help me find it using search engines.",
        "Provide me some keywords of this webpage.",
        "What may be keywords for this webpage?",
        "Summarize the webpage as some keywords."
    ]
}

icon_mixed = {
    "icon_grounding": [
        "I'll provide textual descriptions of float icons in this webpage screenshot. Can you find their coordinates?",
        "From the given webpage screenshot, I need you to identify the locations of described float icons.",
        "Based on this screenshot, I'll describe some icons. Please pinpoint their exact locations.",
        "For the icons I describe in this page capture, can you predict their positions?",
        "I will describe icons from the webpage screenshot; your role is to locate them.",
        "Using the attached screenshot of the webpage, please find the coordinates of described icons.",
        "From the image of this webpage, I will describe icons for you to locate.",
        "I'll give descriptions of certain webpage float icons; please identify where they are in this screenshot.",
        "On this webpage screenshot, I will point out some icons; please predict their exact coordinates.",
        "In this web page image, please locate the float icons as I describe them.",
        "Given this screenshot of a webpage, I'll describe some float icons; locate them for me.",
        "Please use the provided webpage screenshot to locate the icons I describe.",
    ],
    "icon_referring": [
        "Given this screenshot of a webpage, I specify locations of some float icons; please describe these icons there.",
        "From the given webpage image, I'll provide coordinates of some icons; can you describe their main content or role in the webpage at these points?",
        "On this captured webpage, I provide specific image coordinates; you need to predict the content of the icons there.",
        "Using this web page screenshot, I'll indicate some float icons' locations; please summarize these icons in a few words.",
        "With this image of a webpage, I specify coordinates of float icons; your task is to generate a brief description.",
        "In this webpage capture, I'll give coordinates of some icons; can you briefly introduce their content or function?",
        "Based on this screenshot, I provide a series of coordinates of some icons; generate a caption of each one.",
    ],
    "icon_all_grounding": [
        "Find all inserted icons with grounding and describe them.",
        "What and where are the floating icons in the screenshot?",
        "There are several icons pasted in the webpage screenshot. Please point out all of them.",
        "Can you tell me the position and content of all inserted icons in the screen?"
    ]
}

som = [
    "An element is marked with a bounding box. What's the text and the position of it?",
    "Tell me the text content and the position (with grounding) of the element enclosed by the striking rectangular box.",
    "What's the text and coordinates of the marked element?",
    "Point out the precise location and of the rectangular box and caption the element within it.",
    "What is being framed in the screen? What's the content of it? What's the position of it?",
    "Observe the region enclosed by the rectangular box. Generate a brief caption of it and point out its position with grounding.",
    "Where is and what is in the striking bounding box?",
    "Where is the element marked by the striking box and what's the content of it?",
    "Locate the rectangular bounding box with grounding and describe its content briefly."
]

icon_description = [
    "How to describe this icon?",
    "Can you describe this icon briefly?",
    "Describe this icon from the perspectives of content, style, clicking effect, etc.",
    "What does this icon image contain? What style is it? If it is clickable, what effect might clicking it have?",
    "This is a common icon in a webpage or mobile app. How should I describe it?"
]


rico_tasks = {
    "widget-grounding": [
        "In this UI screenshot, what is the position of the element corresponding to the command \"{instruction}\"?",
        "In the UI, where should I click if I want to complete instruction \"{instruction}\"?",
        "In this screen, how can I navigate to the section that says \"{instruction}\"?",
        "On this page, what is the location of the button do I press to follow the command \"{instruction}\"?",
        "For the action described as \"{instruction}\", where is the corresponding icon in this UI?",
        "To execute the function \"{instruction}\", which item in the UI should I select (in coordinates)?",
        "In this UI layout, where is the tool that performs the operation \"{instruction}\"?",
        "On this screen, where can I find the feature that allows me to \"{instruction}\"?",
        "In the software interface, which menu item corresponds to the task \"{instruction}\" (in coordinates)?",
        "Within this dashboard, which widget should I interact with to \"{instruction}\"?",
        "In the UI here, I need to {instruction}, what is the coordinates of the element is related to this?",
        "If my goal is to \"{instruction}\", which control in this interface should I use?",
        "On this device screen, to achieve the outcome \"{instruction}\", where do I tap?",
        "Facing this interface, where do I access to \"{instruction}\"?",
        "In this digital interface, to initiate \"{instruction}\", where is my point of interest?",
        "When using this app, for the function \"{instruction}\", where is the command located?",
        "In this UI design, to process the instruction \"{instruction}\", where should I activate?",
        "Within this graphical user interface, to \"{instruction}\", which icon should I be looking for?",
        "On this web page, to perform \"{instruction}\", where is the link or button I will click?",
        "In this interface snapshot, to begin \"{instruction}\", what is the clicking point?",
        "When interacting with this UI, for the operation labeled \"{instruction}\", what is my target?",
        "On this software's interface, to execute the step \"{instruction}\", where do I direct my attention?",
        "In the current UI, I want to {instruction}, where should I click?",
        "In this image, I want to {instruction}, where should I click on?",
        "In the current UI, to {instruction}, where should I click?",
        "In this image, to {instruction}, where should I click on?",
        "On this screen, I need to {instruction}, where do I click?",
        "In the UI right now, to {instruction}, where should I click?",
        "In this layout, I want to {instruction}, where is the upload button?",
        "On this interface, to {instruction}, where should I click?",
        "In this view, I need to {instruction}, which icon do I select (in coordinates)?",
        "On this page, I want to {instruction}, where is the option?",
        "In this webpage, I'm trying to {instruction}, where do I click?",
        "In this software, to {instruction}, where should I navigate?"
    ],
    "widget-caption": [
        "Please generate a description for the element at {bbox}.",
        "Describe the function of the element at {bbox} on the screen.",
        "What is the function of the element at {bbox} on the UI?",
        "What happens when you tap position {bbox} on the screen?",
        "What happens when you click point {bbox} on the screen?",
        "Can you explain what the user interface element at {bbox} does?",
        "What action is triggered by interacting with the area at {bbox}?",
        "Explain the purpose of the interactive element found at {bbox}.",
        "What feature is accessed by selecting the location at {bbox}?",
        "Identify and describe the component located at {bbox}.",
        "What is the outcome of selecting the element at {bbox}?",
        "Detail the functionality of the UI element positioned at {bbox}.",
        "What is the significance of the element located at {bbox} in the application?",
        "How does the element at {bbox} contribute to the overall user experience?",
        "What kind of input or interaction is expected at the point marked {bbox}?",
        "Carefully read the screen, and then generate a short caption of the element in the region {bbox}.",
        "Can you help me briefly summarize the content or function of the widget {bbox} in the screenshot?",
        "There is a widget in the area {bbox} on the screen, please describe its content in one sentence."
    ],
    "screen2words": [
        "Can you provide a detailed description of the interface screenshot shown?",
        "Illustrate the details visible in the provided screenshot.",
        "What does the presented screen image depict?",
        "How would you narrate the contents of this screen capture to someone who can't see it?",
        "Please detail the elements shown in the interface screenshot.",
        "Describe the features and information displayed in this screenshot.",
        "Elaborate on what is visible in the screenshot of the interface.",
        "Give a comprehensive description of the screenshot's interface.",
        "What information is conveyed in the screenshot displayed?",
        "Could you depict the content and layout of the screen image provided?",
        "Explain the visual aspects of the screenshot taken from this interface.",
        "How would you verbally depict the interface shown in the screenshot?",
        "What key elements are shown in this interface screenshot?",
        "Provide a verbal representation of the screenshot's content.",
        "Narrate the components and information visible in this interface capture.",
        "What are the main features displayed in the screenshot of this screen?",
        "Outline the specific details shown in the interface image.",
        "How would you describe this screen image to someone who cannot see it?",
        "Enumerate the elements and information present in the provided interface screenshot.",
        "Detail the visual composition of the screen capture you see.",
        "Carefully read the screen, and then generate a short caption of its content.",
        "Can you help me briefly summarize the screenshot in one sentence?",
        "When referring to this mobile phone screenshot, how to describe its content in one sentence?"    
    ]
}


advanced_tasks = {
    "detail": [
        "Describe this screen in an exhaustively detailed manner.",
        "Observe this webpage carefully, and provide a description about the count, position and relative position of elements as detailed as possible.",
        "Generate a comprehensive and detailed description of this webpage.",
        "Can you walk us through the details of this webpage, highlighting how many different elements you see and where they are located?",
        "Take a close look at this screen and share your observations about the various elements and how they are arranged in relation to each other.",
        "Describe what you see on this webpage, focusing on the distinct components and their placements within the layout.",
        "What stands out to you in this screenshot? Please detail the elements present and how they relate to one another in the overall design.",
        "As you examine this webpage, what can you tell us about the layout and arrangement of the different features?",
        "Offer a description of this screen that captures the essence of the visible elements and their relative placements.",
        "How would you describe this webpage? Share your thoughts on the arrangement of elements and any interesting details you notice.",
        "Explore this screenshot and tell us about the various components and their interactions within the space.",
        "Provide a thorough breakdown of this webpage, focusing on the total number of visible elements, their specific locations, and how they relate to one another.",
        "Analyze this screen in detail, detailing the quantity and positioning of elements, as well as their spatial relationships.",
        "Give a meticulous account of the elements present on this webpage, including their numbers, locations, and how they are arranged in relation to each other.",
        "Carefully examine this webpage and generate an in-depth description that covers the count, specific placements, and relative distances between elements.",
        "Detail the various components visible on this screen, focusing on their numbers, precise locations, and how they are positioned relative to one another.",
        "Describe this webpage with precision, emphasizing the total number of elements, their exact locations, and their arrangement in relation to other elements.",
        "Examine the content of this screen thoroughly and provide a detailed description of the elements, including their count, positions, and relative placements.",
        "Generate a detailed account of this webpage, including the number of elements, their specific locations, and the relative positioning of each component.",
    ],
    "function": [
        "Summarize the purpose of this screen in one sentence.",
        "What's the function of the page when a user interacts with it?",
        "From the user's perspective, what is the purpose or function of this webpage?",
        "In a single sentence, what key functionality does this webpage offer to users?",
        "How would you define the main purpose of this screen from the user's standpoint?",
        "What is the primary function of this webpage when users engage with it?",
        "Can you describe, in one sentence, how this page serves its users?",
        "From the perspective of a user, what does this screen aim to accomplish?",
        "Briefly outline the core functionality provided by this webpage to its visitors.",
        "What does this page enable users to do, summed up in one concise sentence?",
        "In your own words, what is the main intent of this screen for its users?",
        "In just one sentence, explain the main action users can take on this webpage.",
        "What unique feature does this page provide that enhances user experience, described in a single sentence?",
        "How would you summarize the overall utility of this screen for users?",
        "Describe, in one concise sentence, how this webpage supports users in achieving their goals.",
        "From a user's perspective, what is the essential service offered by this webpage?",
        "What is the primary takeaway regarding the functionality of this screen, captured in one sentence?",
        "Can you succinctly outline the key purpose of this webpage for its users?",
        "In a single sentence, what benefits does this page provide to someone interacting with it?",
    ]
}

monkey_training = [
    "Generate a detailed caption of this image.",
    "Describe this image in detail.",
    "What is in the image?",
    "Can you provide a detailed description of the picture?"
]