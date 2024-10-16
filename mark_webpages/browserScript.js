const vw = Math.max(
  document.documentElement.clientWidth || 0,
  window.innerWidth || 0
);
const vh = Math.max(
  document.documentElement.clientHeight || 0,
  window.innerHeight || 0
);

const elementDataMap = new Map();
let rootElementData = null;
let markedElements = null;

function assert(condition, message) {
  if (!condition) {
    console.log(message);
    throw new Error(message || "Assertion failed");
  }
}

function isDefined(value) {
  return typeof value !== "undefined" && typeof value !== "null";
}

class BoundingBox {
  constructor({ left, top, right, bottom }, clamp = false) {
    assert(
      isDefined(left) && isDefined(top) && isDefined(right) && isDefined(bottom)
    );

    if (clamp) {
      left = BoundingBox.clamp(left, 0, vw);
      top = BoundingBox.clamp(top, 0, vh);
      right = BoundingBox.clamp(right, 0, vw);
      bottom = BoundingBox.clamp(bottom, 0, vh);
    }

    this.left = left;
    this.top = top;
    this.right = right;
    this.bottom = bottom;
    this.width = Math.max(0, this.right - this.left);
    this.height = Math.max(0, this.bottom - this.top);
    this.centerX = this.left + this.width / 2;
    this.centerY = this.top + this.height / 2;
    this.area = this.width * this.height;
    this.clamped = clamp;
  }

  static clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  static centerL1Distance(rect1, rect2) {
    return (
      Math.abs(rect1.centerX - rect2.centerX) +
      Math.abs(rect1.centerY - rect2.centerY)
    );
  }

  static aggregateRects(rects) {
    assert(rects.every((rect) => rect instanceof BoundingBox && rect.clamped));
    if (rects.length === 0) return null;
    let left = vw,
      top = vh,
      right = 0,
      bottom = 0;
    rects
      .filter((rect) => rect.area > 0)
      .forEach((rect) => {
        left = Math.min(left, rect.left);
        top = Math.min(top, rect.top);
        right = Math.max(right, rect.right);
        bottom = Math.max(bottom, rect.bottom);
      });
    return new BoundingBox({ left, top, right, bottom }, (clamp = true));
  }

  crop(rectBase) {
    assert(this.clamped && rectBase instanceof BoundingBox && rectBase.clamped);
    this.left = Math.max(this.left, rectBase.left);
    this.top = Math.max(this.top, rectBase.top);
    this.right = Math.min(this.right, rectBase.right);
    this.bottom = Math.min(this.bottom, rectBase.bottom);
    this.width = Math.max(0, this.right - this.left);
    this.height = Math.max(0, this.bottom - this.top);
    this.centerX = this.left + this.width / 2;
    this.centerY = this.top + this.height / 2;
    this.area = this.width * this.height;
  }
}

class ElementData {
  constructor(element) {
    assert(isDefined(element), "Element is required parameters.");
    this.element = element;
    this.computedStyle = null;

    this.selfRectOri = null;
    this.selfRectVis = null;

    this.selfVisible = null;
    this.visible = null;

    this.selfTypes = new Set();
    this.childrenTypes = new Set();
    this.types = new Set();
    this.asLeaf = false;

    this.reprString = null;
    this.rect = null;
    this.children = [];

    this.text = "";
    this.ariaLabel = null;
    this.title = null;
  }
}

function removeUnwantedElements() {
  // 定义选择器数组，包含所有与渲染无关的元素类型
  const selectors = [
    "script",
    "noscript",
    "iframe",
    'link:not([rel="stylesheet"])',
  ];

  // 遍历选择器并删除匹配的元素
  selectors.forEach((selector) => {
    const elements = document.body.querySelectorAll(selector);
    elements.forEach((element) => element.remove());
  });

  // 删除注释节点
  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_COMMENT,
    null,
    false
  );
  let comment;
  const commentsToRemove = [];
  while ((comment = walker.nextNode())) {
    commentsToRemove.push(comment);
  }
  commentsToRemove.forEach((comment) =>
    comment.parentNode.removeChild(comment)
  );
}

function getElementBoundingRect(element) {
  const elementData = elementDataMap.get(element);
  if (elementData.selfRectOri !== null && elementData.selfRectVis !== null) {
    return {
      selfRectOri: elementData.selfRectOri,
      selfRectVis: elementData.selfRectVis,
    };
  }

  const selfRectOri = new BoundingBox(element.getBoundingClientRect());
  const selfRectVis = new BoundingBox(selfRectOri, (clamp = true));
  elementData.selfRectOri = selfRectOri;
  elementData.selfRectVis = selfRectVis;
  return { selfRectOri: selfRectOri, selfRectVis: selfRectVis };
}

function getTextBoundingRect(element) {
  // 专门针对文本的Bouding Box计算，排除文字很短但是元素占一整行的情况
  if (!element || !(element instanceof Element)) {
    throw new Error("Invalid element provided");
  }
  const range = document.createRange();
  range.selectNodeContents(element);
  const rects = Array.from(range.getClientRects()).map(
    (rect) => new BoundingBox(rect, (clamp = true))
  );
  return BoundingBox.aggregateRects(rects);
}

function isElementSelfVisible(element) {
  const elementData = elementDataMap.get(element);
  if (elementData.selfVisible !== null) {
    return elementData.selfVisible;
  }

  const minAreaThres = 40,
    minWHThres = 5,
    smallAreaThres = 400,
    largeAreaThres = (vh * vw) / 20;
  const visibleRatioThres = 0.5;

  const { selfRectOri, selfRectVis } = getElementBoundingRect(element);
  const selfAreaOri = selfRectOri.area,
    selfAreaVis = selfRectVis.area;
  const computedStyle = window.getComputedStyle(element);
  elementData.computedStyle = {
    hidden:
      computedStyle.opacity === "0" ||
      computedStyle.visibility === "hidden" ||
      computedStyle.display === "none",
    display: computedStyle.display,
    cursor: computedStyle.cursor,
    overflow: computedStyle.overflow,
  };

  // 开发者设置为不可见的情况：computedStyle上被隐藏，或宽高其一为0（总面积为0）
  if (elementData.computedStyle.hidden || selfAreaOri === 0) {
    elementData.selfVisible = false;
    return false;
  }

  // 中心点+角落4个点访问不到该元素，视为不可见（例外：该元素可见面积很大）
  const centerX = selfRectVis.centerX;
  const centerY = selfRectVis.centerY;
  const offsetX = (selfRectVis.width * 0.85) / 2;
  const offsetY = (selfRectVis.height * 0.85) / 2;
  const sampledPoints = [
    { x: centerX, y: centerY }, // center
    { x: centerX - offsetX, y: centerY - offsetY }, // top-left
    { x: centerX + offsetX, y: centerY - offsetY }, // top-right
    { x: centerX - offsetX, y: centerY + offsetY }, // bottom-left
    { x: centerX + offsetX, y: centerY + offsetY }, // bottom-right
  ];
  const accessible = sampledPoints.some((point) => {
    const elAtPoint = document.elementFromPoint(point.x, point.y);
    return (
      elAtPoint === element ||
      element.contains(elAtPoint) ||
      element.parentNode === elAtPoint
    );
  });
  if (!accessible && selfAreaVis < largeAreaThres) {
    elementData.selfVisible = false;
    return false;
  }

  // 计算可见比例（总面积为0的情况已经提前返回）
  const visibleRatio = selfAreaVis / selfAreaOri;

  // 最终判断：全可见且面积不过小，或可见比例够大且面积大于阈值，或可见比例不够大但显示面积足够大
  let selfVisible = false;
  if (
    (visibleRatio === 1 &&
      selfAreaVis >= minAreaThres &&
      Math.min(selfRectVis.width, selfRectVis.height) > minWHThres) ||
    (visibleRatio >= visibleRatioThres && selfAreaVis >= smallAreaThres) ||
    (visibleRatio < visibleRatioThres && selfAreaVis >= largeAreaThres)
  ) {
    selfVisible = true;
  }

  elementData.selfVisible = selfVisible;
  return selfVisible;
}

function traverseNodeVisibility(element) {
  if (!element) return;

  // 为每个元素初始化Info类
  const elementData = new ElementData(element);
  elementDataMap.set(element, elementData);

  // 判断自身可见性
  const selfVisible = isElementSelfVisible(element);
  // 递归子节点
  const childrenVisible = Array.from(element.children)
    .map((child) => traverseNodeVisibility(child))
    .some((childVisible) => childVisible === true);

  // 如果该节点的overflow是hidden，则裁剪所有其子节点的边界框
  if (elementData.computedStyle.overflow === "hidden") {
    Array.from(element.children).forEach((child) =>
      elementDataMap.get(child).selfRectVis.crop(elementData.selfRectVis)
    );
  }

  // 如果不是被刻意隐藏，则最终可见性= 自身可见 or (子节点溢出可展示 and 子节点可见)
  const visible =
    !elementData.computedStyle.hidden &&
    (selfVisible ||
      (elementData.computedStyle.overflow === "visible" && childrenVisible));
  elementData.visible = visible;

  return visible;
}

class Types {
  static TEXT = Symbol("Text");
  static CODE = Symbol("Code");

  static IMAGE = Symbol("Image");
  static ICON = Symbol("Icon");

  static BUTTON = Symbol("Button");
  static HREF = Symbol("Href");
  static INPUTBOX = Symbol("InputBox");
  static SELECTBOX = Symbol("SelectBox");
  static CHECKBOX = Symbol("CheckBox");
  static COMBOBOX = Symbol("ComboBox");
  static VIDEO = Symbol("Video");
  static INTERACTIVE = Symbol("Interactive");
  static CLICKABLE = Symbol("Clickable");

  static #vaildTypes = new Set(
    Object.values(this).filter((value) => typeof value === "symbol")
  );

  static isValid(type) {
    return this.#vaildTypes.has(type);
  }
}

class TypeUtils {
  static getCodeType(element) {
    return element.tagName === "CODE" ? Types.CODE : null;
  }

  static imageTagNames = new Set(["IMG", "IMAGE", "I", "SVG", "CANVAS"]);
  static getImageType(element) {
    const tagName = element.tagName.toUpperCase();
    if (!this.imageTagNames.has(tagName)) return null;
    const rect = elementDataMap.get(element).selfRectVis;
    if (tagName == "IMG" || tagName == "IMAGE") {
      const whRatio = rect.width / rect.height;
      return whRatio > 0.8 && whRatio < 1.2 && rect.area <= 2500
        ? Types.ICON
        : Types.IMAGE;
    } else if (tagName == "SVG" || tagName == "I") {
      return Types.ICON;
    } else {
      return rect.area > 10000 ? Types.IMAGE : Types.ICON;
    }
  }

  static getInteractiveType(element) {
    // TODO: 利用role属性标注更多内容，示例：github
    switch (element.role) {
      case "button":
        return Types.BUTTON;
      case "checkbox":
        return Types.CHECKBOX;
      case "combobox":
        return Types.COMBOBOX;
      case "textbox":
        return Types.INPUTBOX;
    }
    switch (element.tagName) {
      case "INPUT":
        switch (element.type) {
          case "submit":
          case "file":
          case "reset":
          case "button":
            return Types.BUTTON;
          case "text":
          case "number":
          case "password":
          case "email":
          case "tel":
          case "search":
            return Types.INPUTBOX;
          default:
            return Types.INTERACTIVE;
        }
      case "BUTTON":
        return Types.BUTTON;
      case "TEXTAREA":
        return Types.INPUTBOX;
      case "A":
        if (element.href.trim().length > 0) return Types.HREF;
        break;
      case "SELECT":
        return Types.SELECTBOX;
      case "VIDEO":
        return Types.VIDEO;
      case "IFRAME":
      case "DETAILS":
      case "SUMMARY":
        return Types.INTERACTIVE;
    }
    if (elementDataMap.get(element).computedStyle.cursor === "pointer")
      return Types.CLICKABLE;
    return null;
  }

  static setCT = new Set([Types.CODE, Types.TEXT]);
  static setHCT = new Set([Types.HREF, Types.CODE, Types.TEXT]);
  static simplifyAsLeaf(element, selfTypes, types) {
    // 是图片，删除文本类型，提前返回
    function isImageType(selfTypes, types) {
      if (selfTypes.has(Types.ICON) || selfTypes.has(Types.IMAGE)) {
        types.delete(Types.TEXT);
        return true;
      }
      return false;
    }

    // 是代码或一段代码，删除Text类型，提前返回
    function isCodeType(element, selfTypes, types) {
      if (
        selfTypes.has(Types.CODE) ||
        (element.tagName === "PRE" &&
          types.has(Types.CODE) &&
          types.isSubsetOf(TypeUtils.setCT))
      ) {
        types.delete(Types.TEXT);
        return true;
      }
      return false;
    }

    // 自身是超链接/按钮，删除其他冗余类型，提前返回
    function isHrefOrButtonType(selfTypes, types) {
      if (selfTypes.has(Types.HREF) || selfTypes.has(Types.BUTTON)) {
        types.delete(Types.CLICKABLE);

        const selfType = selfTypes.has(Types.HREF) ? Types.HREF : Types.BUTTON;
        // Href/Button + Icon + Text，视为Href/Button，提前返回
        if (types.isSubsetOf(new Set([selfType, Types.ICON, Types.TEXT]))) {
          types.clear();
          types.add(selfType);
          return true;
        }
      }
      return false;
    }

    function isTextType(element, selfTypes, types) {
      // 纯文本节点（只有Text类型，且子元素都是inline），selfTypes添加Text，提前返回
      // 自身不是Href，Href + Text，且子元素自身都是Text或Href且都是inline，视为纯Text，提前返回
      const children = Array.from(element.children).filter(
        (child) => elementDataMap.get(child).visible
      );
      if (
        selfTypes.size === 0 &&
        types.has(Types.TEXT) &&
        types.isSubsetOf(TypeUtils.setHCT) &&
        children.every((child) => {
          const childData = elementDataMap.get(child);
          const childSelfTypes = childData.selfTypes;
          const childDisplay = childData.computedStyle.display;
          return (
            childSelfTypes.isSubsetOf(TypeUtils.setHCT) &&
            ["inline", "inline-block", "inline-flex"].includes(childDisplay)
          );
        })
      ) {
        types.delete(Types.HREF);
        types.delete(Types.CODE);
        selfTypes.add(Types.TEXT);
        return true;
      }
      return false;
    }

    return (
      isImageType(selfTypes, types) ||
      isCodeType(element, selfTypes, types) ||
      isHrefOrButtonType(selfTypes, types) ||
      isTextType(element, selfTypes, types)
    );
  }
}

function traverseNodeTypes(node) {
  // 文本节点，直接返回
  if (node.nodeType === Node.TEXT_NODE) {
    return node.nodeValue.trim().length > 0 ? new Set([Types.TEXT]) : new Set();
  }

  // 可见性检查
  assert(node.nodeType === Node.ELEMENT_NODE, node);
  const elementData = elementDataMap.get(node);
  if (!elementData.visible) return new Set();

  // 获取selfTypes
  const selfTypes = new Set();
  const codeType = TypeUtils.getCodeType(node);
  const imageType = TypeUtils.getImageType(node);
  const interactiveType = TypeUtils.getInteractiveType(node);
  if (Types.isValid(codeType)) selfTypes.add(codeType);
  if (Types.isValid(imageType)) selfTypes.add(imageType);
  if (Types.isValid(interactiveType)) selfTypes.add(interactiveType);

  // 递归
  const childrenTypes = new Set();
  Array.from(node.childNodes).forEach((childNode) => {
    const childTypes = traverseNodeTypes(childNode);
    childTypes.forEach((type) => childrenTypes.add(type));
  });

  // 整合：简化和设置停止展开
  const types = new Set([...selfTypes, ...childrenTypes]);
  elementData.asLeaf = TypeUtils.simplifyAsLeaf(node, selfTypes, types);

  // 如果是纯文本，重新计算边界框
  if (selfTypes.size === 1 && selfTypes.has(Types.TEXT))
    elementData.selfRectVis = getTextBoundingRect(node);

  // 赋值
  selfTypes.forEach((type) => elementData.selfTypes.add(type));
  childrenTypes.forEach((type) => elementData.childrenTypes.add(type));
  types.forEach((type) => elementData.types.add(type));
  return types;
}

function traverseElementData(element) {
  const elementData = elementDataMap.get(element);
  if (elementData.types.size === 0) return elementData;

  const childrenRects = [];
  Array.from(element.children).forEach((child) => {
    const childData = traverseElementData(child);
    if (childData.types.size > 0) {
      elementData.children.push(childData);
      childrenRects.push(childData.rect);
    }
  });
  elementData.reprString = `${element.tagName}[${elementData.children
    .map((childData) => childData.reprString)
    .join(",")}]`;

  // 聚合边界框
  let rect = getElementBoundingRect(element).selfRectVis;
  if (elementData.computedStyle.overflow === "visible") {
    rect = BoundingBox.aggregateRects([rect].concat(childrenRects));
  }
  elementData.rect = rect;

  // 记录文本（根据元素不同，取的属性不同）
  const selfTypes = elementData.selfTypes;
  let elementText = "";
  if (selfTypes.has(Types.IMAGE) || selfTypes.has(Types.ICON)) {
    elementText = element.alt ?? "";
  } else if (selfTypes.has(Types.BUTTON) && element.tagName === "INPUT") {
    elementText = element.value ?? "";
  } else if (selfTypes.has(Types.INPUTBOX)) {
    elementText = (element.placeholder ?? "") + (element.value ?? "");
  } else if (selfTypes.has(Types.SELECTBOX)) {
    elementText = element.options[element.selectedIndex].text ?? "";
  } else if (selfTypes.size > 0) {
    const text =
      "innerText" in element ? element.innerText : element.textContent;
    elementText = text.trim().replace(/\s{2,}/g, " ") ?? "";
  }
  if (elementText.length === 0) {
    elementData.children.forEach((childData) => {
      if (childData.text.length > 0) elementText += childData.text + "\n";
    });
  }
  elementData.text = elementText.slice(0, 1000);
  elementData.ariaLabel = element.ariaLabel ?? "";
  elementData.title = element.title ?? "";

  return elementData;
}

function findNodesAtDepth(node, depth) {
  if (depth === 0 || node.children.length === 0 || node.asLeaf) {
    return [node];
  }

  let nodes = [];
  node.children.forEach((child) => {
    nodes = nodes.concat(findNodesAtDepth(child, depth - 1));
  });

  return nodes;
}

function getRandomColor() {
  let color = "#";
  for (let i = 0; i < 6; i++) {
    color += Math.floor(Math.random() * 16).toString(16);
  }
  return color;
}

function markElements(elementDataList, markTypes) {
  const bbox_label_group = document.createElement("div");
  bbox_label_group.id = "__marked_group_109938";
  document.body.appendChild(bbox_label_group);

  elementDataList.forEach(function (elementData, index) {
    const mark = document.createElement("div");
    const borderColor = getRandomColor();
    mark.style.outline = `2px dashed ${borderColor}`;
    mark.style.position = "fixed";
    mark.style.left = elementData.rect.left + "px";
    mark.style.top = elementData.rect.top + "px";
    mark.style.width = elementData.rect.width + "px";
    mark.style.height = elementData.rect.height + "px";
    mark.style.pointerEvents = "none";
    mark.style.boxSizing = "border-box";
    mark.style.zIndex = 2147483647;
    mark.style.backgroundColor = "transparent";

    // Add floating label at the corner
    const label = document.createElement("span");
    const types = Array.from(elementData.types).map((type) => type.description);
    label.textContent = markTypes ? `${index}: ${types.join(" ")}` : `${index}`;
    label.style.position = "absolute";
    if (elementData.rect.top <= 30) {
      label.style.top = mark.style.height;
      label.style.left = mark.style.width;
    } else {
      label.style.top = "-30px";
      label.style.left = "0px";
    }
    label.style.background = borderColor;
    label.style.color = "white";
    label.style.padding = "1px 1px";
    label.style.fontSize = markTypes ? "12px" : "17px";
    label.style.borderRadius = "1px";
    mark.appendChild(label);

    bbox_label_group.appendChild(mark);
  });
}

function postProcess(elementData) {
  return {
    tagName: elementData.element.tagName,
    types: Array.from(elementData.types).map((type) => type.description),
    text: elementData.text,
    bbox: [
      Math.round(elementData.rect.left),
      Math.round(elementData.rect.top),
      Math.round(elementData.rect.right),
      Math.round(elementData.rect.bottom),
    ],
    ariaLabel: elementData.ariaLabel,
    title: elementData.title,
  };
}

function postProcessArray(elementDataList) {
  return elementDataList.map((elementData) => postProcess(elementData));
}

function pruneTree(root) {
  function singleChildPrune(node, parent, parIndex) {
    if (node.children.length === 0) return;
    node.children.forEach((child, index) =>
      singleChildPrune(child, node, index)
    );

    // 单子节点，进入截短判断
    if (node.children.length === 1) {
      const child = node.children[0];
      const isAllDIV =
        node.element.tagName === "DIV" && child.element.tagName === "DIV";
      // 必须父与子同时是div，且不是根节点，才继续判断
      if (parent === null || !isAllDIV) return;

      const nRect = node.rect;
      const nArea = nRect.area;
      const cRect = child.rect;
      const cArea = cRect.area;
      // 1. 自身面积为零且没有selfTypes，去父留子
      if (node.selfRectVis.area === 0 && node.selfTypes.size === 0) {
        parent.children[parIndex] = child;
        // 2. 父与子text内容、types、中心都相同，且面积差小
      } else if (
        node.text === child.text &&
        node.types.size === child.types.size &&
        child.types.isSubsetOf(node.types) &&
        BoundingBox.centerL1Distance(nRect, cRect) < 5 &&
        (cArea / nArea > 0.6 || (cArea / nArea > 0.4 && nArea - cArea < 1500))
      ) {
        // （1）面积差距很小，去子留父（保留更大的框）
        if (nArea - cArea < 1500) {
          node.children = child.children;
          // （2）面积差距较大，去父留子（提高容错率）
        } else {
          parent.children[parIndex] = child;
        }
      }
    }
  }
  singleChildPrune(root, null, null);
  return root;
}

function aggregateElementData() {
  // 0. 清空原信息
  elementDataMap.clear();
  rootElementData = null;
  markedElements = null;

  // 1. 删除渲染无关元素
  removeUnwantedElements();

  // 2. 第一次遍历，确定元素可见性
  traverseNodeVisibility(document.body);

  // 3. 第二次遍历，确定元素类别
  traverseNodeTypes(document.body);

  // 4. 第三次遍历，整合元素信息
  rootElementData = traverseElementData(document.body);

  // 5. 第四次遍历，裁剪树深
  rootElementData = pruneTree(rootElementData);
}

function _unmarkPage() {
  const group = document.getElementById("__marked_group_109938");
  if (group) group.remove();
  markedElements = null;
}

function _markPage(depth = 99, force_reaggregate = false, markTypes = true) {
  _unmarkPage();

  // 整合信息
  if (rootElementData === null || force_reaggregate) aggregateElementData();

  // 展示
  let annotations = null;
  if (rootElementData.children.length > 0 && rootElementData.types.size > 0) {
    let elementsToMark = null;
    if (depth >= 0) {
      elementsToMark = findNodesAtDepth(rootElementData, depth);
      markElements(elementsToMark, markTypes);
      markedElements = elementsToMark;
      annotations = postProcessArray(elementsToMark);
    } else {
      throw Error(`Argument "depth" must be positive, not ${depth}!`);
    }
  }
  return annotations;
}
