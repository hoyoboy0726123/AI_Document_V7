import { useCallback, useEffect, useRef, useState } from "react";
import "./FolderTree.css";
import { Button, Dropdown, Input, Modal, Tree, Typography, message } from "antd";
import {
  DeleteOutlined,
  EditOutlined,
  FolderAddOutlined,
  FolderFilled,
  FolderOpenFilled,
  PlusOutlined,
} from "@ant-design/icons";
import { useNavigate, useSearchParams } from "react-router-dom";
import apiClient from "../../services/api";

const { DirectoryTree } = Tree;

// Compute recursive doc count (folder itself + all descendants)
function computeRecursiveCount(node) {
  const childSum = (node.children || []).reduce(
    (sum, child) => sum + computeRecursiveCount(child),
    0
  );
  node.recursiveCount = node.docCount + childSum;
  return node.recursiveCount;
}

// Build Ant Design Tree treeData from flat folder list
function buildTree(folders, totalCount = 0, unassignedCount = 0) {
  const map = {};
  const roots = [];

  folders.forEach((f) => {
    map[f.id] = {
      key: f.id,
      title: f.name,
      folder: f,
      docCount: f.doc_count || 0,
      recursiveCount: f.doc_count || 0,
      children: [],
      isLeaf: false,
      selectable: true,
    };
  });

  folders.forEach((f) => {
    if (f.parent_id && map[f.parent_id]) {
      map[f.parent_id].children.push(map[f.id]);
    } else {
      roots.push(map[f.id]);
    }
  });

  // Sort children by order_index then name
  const sortNodes = (nodes) => {
    nodes.sort((a, b) => {
      const oi = (a.folder?.order_index ?? 0) - (b.folder?.order_index ?? 0);
      return oi !== 0 ? oi : a.title.localeCompare(b.title);
    });
    nodes.forEach((n) => n.children && sortNodes(n.children));
  };
  sortNodes(roots);

  // Bottom-up pass: compute recursive counts
  roots.forEach((r) => computeRecursiveCount(r));

  const allNode = {
    key: "__all__",
    title: "全部文件",
    docCount: totalCount,
    isLeaf: false,
    isVirtual: true,
    selectable: true,
  };
  const unclassNode = {
    key: "__root__",
    title: "未歸類",
    docCount: unassignedCount,
    isLeaf: true,
    isVirtual: true,
    selectable: true,
  };

  return [allNode, ...roots, unclassNode];
}

// Collect all non-virtual keys from tree for auto-expand
function getAllKeys(nodes) {
  const keys = [];
  const walk = (list) => {
    list.forEach((n) => {
      if (!n.isVirtual) keys.push(n.key);
      if (n.children?.length) walk(n.children);
    });
  };
  walk(nodes);
  return keys;
}

const FolderTree = ({ style }) => {
  const [folders, setFolders] = useState([]);
  const [treeData, setTreeData] = useState([]);
  const [expandedKeys, setExpandedKeys] = useState(["__all__"]);
  const [totalCount, setTotalCount] = useState(0);
  const [unassignedCount, setUnassignedCount] = useState(0);
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  // Modal state for create/rename
  const [modalOpen, setModalOpen] = useState(false);
  const [modalTitle, setModalTitle] = useState("");
  const [folderName, setFolderName] = useState("");
  const [editingFolder, setEditingFolder] = useState(null); // null = create
  const [parentId, setParentId] = useState(null);
  const [saving, setSaving] = useState(false);
  const inputRef = useRef(null);

  const currentFolderId = searchParams.get("folder_id") ?? "__all__";

  const fetchFolders = useCallback(async () => {
    try {
      const [foldersResp, totalResp, unassignedResp] = await Promise.all([
        apiClient.get("folders"),
        apiClient.get("documents/", { params: { page: 1, page_size: 1 } }),
        apiClient.get("documents/", { params: { page: 1, page_size: 1, folder_id: "__root__" } }),
      ]);
      setFolders(foldersResp.data ?? []);
      setTotalCount(totalResp.data?.total ?? 0);
      setUnassignedCount(unassignedResp.data?.total ?? 0);
    } catch {
      // silent
    }
  }, []);

  useEffect(() => {
    fetchFolders();
  }, [fetchFolders]);

  useEffect(() => {
    const tree = buildTree(folders, totalCount, unassignedCount);
    setTreeData(tree);
    setExpandedKeys((prev) => {
      const all = new Set([...prev, ...getAllKeys(tree)]);
      return [...all];
    });
  }, [folders, totalCount, unassignedCount]);

  useEffect(() => {
    if (modalOpen) setTimeout(() => inputRef.current?.focus(), 100);
  }, [modalOpen]);

  const selectFolder = (keys) => {
    const key = keys[0];
    if (!key) return;
    if (key === "__all__") {
      setSearchParams({});
    } else {
      setSearchParams({ folder_id: key });
    }
    navigate({ pathname: "/documents", search: key === "__all__" ? "" : `?folder_id=${key}` });
  };

  const openCreate = (pid = null, parentName = null) => {
    setEditingFolder(null);
    setParentId(pid);
    setFolderName("");
    setModalTitle(parentName ? `在「${parentName}」內新增子資料夾` : "新增資料夾");
    setModalOpen(true);
  };

  const openRename = (folder) => {
    setEditingFolder(folder);
    setParentId(null);
    setFolderName(folder.name);
    setModalTitle("重新命名");
    setModalOpen(true);
  };

  const handleSave = async () => {
    if (!folderName.trim()) return;
    setSaving(true);
    try {
      if (editingFolder) {
        await apiClient.put(`folders/${editingFolder.id}`, { name: folderName.trim() });
        message.success("已重新命名");
      } else {
        await apiClient.post("folders", { name: folderName.trim(), parent_id: parentId });
        message.success("資料夾已建立");
      }
      setModalOpen(false);
      fetchFolders();
    } catch (err) {
      message.error(err.response?.data?.detail ?? "操作失敗");
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (folder) => {
    Modal.confirm({
      title: `刪除「${folder.name}」？`,
      content: "子資料夾將移至上層，資料夾內文件將變為未歸類。",
      okText: "刪除",
      okType: "danger",
      cancelText: "取消",
      onOk: async () => {
        try {
          await apiClient.delete(`folders/${folder.id}`);
          message.success("已刪除");
          // If currently viewing this folder, go to all
          if (currentFolderId === folder.id) {
            navigate("/documents");
          }
          fetchFolders();
        } catch (err) {
          message.error(err.response?.data?.detail ?? "刪除失敗");
        }
      },
    });
  };

  // Drag-drop to reorder / reparent
  const onDrop = async ({ node, dragNode, dropToGap }) => {
    if (node.isVirtual || dragNode.isVirtual) return;
    const targetFolder = node.folder;
    const dragFolder = dragNode.folder;

    try {
      if (dropToGap) {
        // Dropped between nodes — move to same parent as target
        await apiClient.put(`folders/${dragFolder.id}`, {
          parent_id: targetFolder.parent_id ?? "__root__",
        });
      } else {
        // Dropped onto a node — make it a child of target
        await apiClient.put(`folders/${dragFolder.id}`, {
          parent_id: targetFolder.id,
        });
      }
      fetchFolders();
    } catch (err) {
      message.error(err.response?.data?.detail ?? "移動失敗");
    }
  };

  const renderTitle = (nodeData) => {
    if (nodeData.isVirtual) {
      return (
        <span style={{ color: "rgba(255,255,255,0.65)", fontSize: 13 }}>
          {nodeData.title}
          {nodeData.docCount > 0 && (
            <span style={{ color: "rgba(255,255,255,0.35)", fontSize: 11, marginLeft: 4 }}>
              ({nodeData.docCount})
            </span>
          )}
        </span>
      );
    }
    const folder = nodeData.folder;
    const items = [
      {
        key: "rename",
        label: "重新命名",
        icon: <EditOutlined />,
        onClick: () => openRename(folder),
      },
      {
        key: "add-sub",
        label: "新增子資料夾",
        icon: <FolderAddOutlined />,
        onClick: () => openCreate(folder.id, folder.name),
      },
      {
        key: "delete",
        label: "刪除",
        icon: <DeleteOutlined />,
        danger: true,
        onClick: () => handleDelete(folder),
      },
    ];

    // Show recursive count (folder itself + all sub-folders)
    const displayCount = nodeData.recursiveCount ?? nodeData.docCount;
    const countLabel =
      displayCount > 0 ? (
        <span style={{ color: "rgba(255,255,255,0.35)", fontSize: 11, marginLeft: 4 }}>
          ({displayCount})
        </span>
      ) : null;

    return (
      <Dropdown menu={{ items }} trigger={["contextMenu"]}>
        <span style={{ color: "rgba(255,255,255,0.85)", fontSize: 13, userSelect: "none" }}>
          {folder.name}
          {countLabel}
        </span>
      </Dropdown>
    );
  };

  const selectedKeys = [currentFolderId];

  return (
    <div className="folder-tree-dark" style={{ ...style }}>
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "8px 16px 4px",
          borderTop: "1px solid rgba(255,255,255,0.08)",
        }}
      >
        <Typography.Text style={{ color: "rgba(255,255,255,0.45)", fontSize: 11, letterSpacing: 1 }}>
          資料夾
        </Typography.Text>
        <Button
          type="text"
          size="small"
          icon={<PlusOutlined />}
          style={{ color: "rgba(255,255,255,0.45)" }}
          onClick={() => {
            // If a real folder is selected, + creates a sub-folder inside it
            if (currentFolderId && currentFolderId !== "__all__" && currentFolderId !== "__root__") {
              const parent = folders.find((f) => f.id === currentFolderId);
              openCreate(currentFolderId, parent?.name ?? null);
            } else {
              openCreate(null);
            }
          }}
          title={
            currentFolderId && currentFolderId !== "__all__" && currentFolderId !== "__root__"
              ? `在選取的資料夾內新增子資料夾`
              : "新增根目錄資料夾"
          }
        />
      </div>

      <DirectoryTree
        treeData={treeData}
        selectedKeys={selectedKeys}
        expandedKeys={expandedKeys}
        onExpand={(keys) => setExpandedKeys(keys)}
        onSelect={selectFolder}
        draggable={(node) => !node.isVirtual}
        onDrop={onDrop}
        titleRender={renderTitle}
        icon={(props) => {
          if (props.isVirtual) return null;
          return props.expanded ? (
            <FolderOpenFilled style={{ color: "#faad14" }} />
          ) : (
            <FolderFilled style={{ color: "#faad14" }} />
          );
        }}
        style={{
          background: "transparent",
          color: "rgba(255,255,255,0.85)",
          fontSize: 13,
          padding: "0 8px",
        }}
        blockNode
      />

      <Modal
        title={modalTitle}
        open={modalOpen}
        onOk={handleSave}
        onCancel={() => setModalOpen(false)}
        okText="儲存"
        cancelText="取消"
        confirmLoading={saving}
      >
        <Input
          ref={inputRef}
          value={folderName}
          onChange={(e) => setFolderName(e.target.value)}
          placeholder="資料夾名稱"
          onPressEnter={handleSave}
          maxLength={64}
        />
      </Modal>
    </div>
  );
};

export default FolderTree;
